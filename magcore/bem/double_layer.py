from __future__ import annotations

import numpy as np

from magcore.bem import singular_quadrature as _ssq
from magcore.bem.adaptive_single_layer import AdaptiveIntegrationConfig
from magcore.bem.element_integrals import triangle_diameter
from magcore.bem.normal_derivative_kernels import laplace_dgreen_dn_y
from magcore.bem.pair_classification import FacePairRelation, face_pair_relation, shared_vertices
from magcore.bem.quadrature import get_triangle_quadrature, triangle_collapsed_gauss
from magcore.bem.spaces import FaceP0Space, VertexP1Space
from magcore.bem.triangle_subdivision import subdivide_triangle_4, triangle_pair_is_regular
from magcore.mesh.surface_mesh import SurfaceMesh


def double_layer_potential_at_points(
    surface_mesh: SurfaceMesh,
    psi_vertex_values: np.ndarray,
    target_points: np.ndarray,
    outward_normals: np.ndarray,
    quadrature_order: int = 2,
) -> np.ndarray:
    """
    Потенциал двойного слоя в точках x ВНЕ Γ (фаза B, docs/math/coupling.md §4):

        (Kψ)(x) = ∮_Γ ∂G(x,y)/∂n_y · ψ(y) ds_y,   ∂G/∂n_y = (x−y)·n_y / (4π|x−y|³),

    где ψ — кусочно-линейный (P1) след на Γ. `outward_normals` — внешние нормали
    граней Γ (соответствуют ориентации surface_mesh; так их выдаёт CouplingInterface).

    Тождество телесного угла (проверка ядра/интегрирования/нормали):
        (K·1)(x) = −1 для x внутри Ω; 0 для x снаружи.
    """
    psi = np.asarray(psi_vertex_values, dtype=float)
    if psi.shape != (surface_mesh.n_vertices,):
        raise ValueError("psi_vertex_values must have shape (n_vertices,).")
    targets = np.asarray(target_points, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("target_points must have shape (M, 3).")
    normals = np.asarray(outward_normals, dtype=float)
    if normals.shape != (surface_mesh.n_faces, 3):
        raise ValueError("outward_normals must have shape (n_faces, 3).")

    q = get_triangle_quadrature(quadrature_order)
    qp = q.points
    qw = q.weights
    lam = np.column_stack([1.0 - qp[:, 0] - qp[:, 1], qp[:, 0], qp[:, 1]])  # (Q, 3) барицентрики

    out = np.zeros(targets.shape[0], dtype=float)
    for f in range(surface_mesh.n_faces):
        tri = surface_mesh.face_vertices(f)
        n_f = normals[f]
        e1 = tri[1] - tri[0]
        e2 = tri[2] - tri[0]
        jac = float(np.linalg.norm(np.cross(e1, e2)))  # = 2 * площадь
        psi_nodes = psi[surface_mesh.faces[f]]
        ys = tri[0] + np.outer(qp[:, 0], e1) + np.outer(qp[:, 1], e2)  # (Q, 3)
        psi_q = lam @ psi_nodes  # (Q,)
        for t in range(targets.shape[0]):
            x = targets[t]
            s = 0.0
            for k in range(qw.shape[0]):
                s += float(qw[k]) * laplace_dgreen_dn_y(x, ys[k], n_f) * float(psi_q[k])
            out[t] += jac * s
    return out


_FOUR_PI = 4.0 * np.pi


def _barycentric_affine_map(tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (v0, P): барицентрики (s,t) точки y (в плоскости tri) = P @ (y - v0), λ=[1-s-t,s,t]."""
    v0 = tri[0]
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    g = np.array([[e1 @ e1, e1 @ e2], [e1 @ e2, e2 @ e2]], dtype=float)
    p = np.linalg.solve(g, np.array([e1, e2], dtype=float))  # (2, 3)
    return v0, p


def _double_layer_pair_regular_vec3(target_tri, source_tri, n_g, v0_orig, p_orig, qpts, qw) -> np.ndarray:
    """∫_{Tt}∫_{Ts} ∂G/∂n_y·[λ0,λ1,λ2](y); λ — барицентрики исходной source-грани (v0_orig, p_orig)."""
    et1 = target_tri[1] - target_tri[0]
    et2 = target_tri[2] - target_tri[0]
    es1 = source_tri[1] - source_tri[0]
    es2 = source_tri[2] - source_tri[0]
    jac = float(np.linalg.norm(np.cross(et1, et2)) * np.linalg.norm(np.cross(es1, es2)))

    xt = target_tri[0] + np.outer(qpts[:, 0], et1) + np.outer(qpts[:, 1], et2)  # (Q, 3)
    ys = source_tri[0] + np.outer(qpts[:, 0], es1) + np.outer(qpts[:, 1], es2)  # (Q, 3)
    st = (ys - v0_orig) @ p_orig.T  # (Qs, 2)
    lam = np.column_stack([1.0 - st[:, 0] - st[:, 1], st[:, 0], st[:, 1]])  # (Qs, 3)

    # Векторизация по полной сетке (Qt×Qs) — без python-цикла по тест-узлам.
    rv = xt[:, None, :] - ys[None, :, :]  # (Qt, Qs, 3)
    r2 = np.einsum("tsj,tsj->ts", rv, rv)
    kern = (rv @ n_g) / (_FOUR_PI * r2 * np.sqrt(r2))  # (Qt, Qs)
    w = qw[:, None] * qw[None, :]  # (Qt, Qs)
    return jac * np.einsum("ts,ts,sk->k", w, kern, lam)


def _double_layer_pair_adaptive_vec3(
    target_tri, source_tri, n_g, v0_orig, p_orig, qpts, qw, near_factor, max_depth, depth=0
) -> np.ndarray:
    """Подразбиение для near/некомпланарных пар; регулярная квадратура, когда пара разрешена или достигнута глубина."""
    if depth >= max_depth or triangle_pair_is_regular(target_tri, source_tri, near_factor=near_factor):
        return _double_layer_pair_regular_vec3(target_tri, source_tri, n_g, v0_orig, p_orig, qpts, qw)

    acc = np.zeros(3, dtype=float)
    if triangle_diameter(target_tri) >= triangle_diameter(source_tri):
        for child in subdivide_triangle_4(target_tri):
            acc += _double_layer_pair_adaptive_vec3(
                child, source_tri, n_g, v0_orig, p_orig, qpts, qw, near_factor, max_depth, depth + 1
            )
    else:
        for child in subdivide_triangle_4(source_tri):
            acc += _double_layer_pair_adaptive_vec3(
                target_tri, child, n_g, v0_orig, p_orig, qpts, qw, near_factor, max_depth, depth + 1
            )
    return acc


def assemble_double_layer_galerkin(
    surface_mesh: SurfaceMesh,
    outward_normals: np.ndarray,
    phi_space: VertexP1Space,
    flux_space: FaceP0Space,
    config: AdaptiveIntegrationConfig | None = None,
) -> np.ndarray:
    """
    Galerkin-матрица двойного слоя K (P0-тест × P1-trial), фаза B (docs/math/coupling.md §4):

        K[f, m] = ∫_{T_f} ∫_Γ ∂G(x,y)/∂n_y · λ_m(y) ds_y ds_x.

    Плоские панели: ядро ∂G/∂n_y тождественно нулевое для компланарных граней
    (включая self) ⇒ такие пары пропускаются. Дальние пары — регулярная двойная
    квадратура; near/некомпланарные смежные — адаптивное подразбиение.

    Возвращает (flux_space.ndofs × phi_space.ndofs). Сопряжённый K' = Kᵀ (Galerkin).
    Проверка Кальдерона: Σ_m K[f,m] = ∫_{T_f} (K·1) ds_x = −A_f/2 (плоская Γ).

    Касающиеся непланарные пары (общее ребро / общая вершина) при
    config.use_sauter_schwab=True считаются полусейминалитической квадратурой
    Заутера–Шваба (O(1) на пару, экспоненциальная сходимость); компланарные пары
    (включая self) дают 0; дальние/near — регулярная/адаптивная квадратура.
    use_sauter_schwab=False восстанавливает прежнее наивное подразбиение.
    """
    if config is None:
        config = AdaptiveIntegrationConfig()

    normals = np.asarray(outward_normals, dtype=float)
    if normals.shape != (surface_mesh.n_faces, 3):
        raise ValueError("outward_normals must have shape (n_faces, 3).")

    qpts = np.asarray(get_triangle_quadrature(config.quadrature_order).points, dtype=float)
    qw = np.asarray(get_triangle_quadrature(config.quadrature_order).weights, dtype=float)
    if config.regular_order > 0:
        _rr = triangle_collapsed_gauss(config.regular_order)
        reg_pts = np.asarray(_rr.points, dtype=float)
        reg_wts = np.asarray(_rr.weights, dtype=float)
    else:
        reg_pts, reg_wts = qpts, qw
    nf = surface_mesh.n_faces
    K = np.zeros((flux_space.ndofs, phi_space.ndofs), dtype=float)

    tris = [surface_mesh.face_vertices(g) for g in range(nf)]
    diams = [triangle_diameter(tris[g]) for g in range(nf)]
    bary = [_barycentric_affine_map(tris[g]) for g in range(nf)]
    face_phi_dofs = [
        [phi_space.vertex_to_dof[int(v)] for v in surface_mesh.faces[g]] for g in range(nf)
    ]

    use_ss = config.use_sauter_schwab

    for fi in range(nf):
        tri_f = tris[fi]
        test_dof = flux_space.face_to_dof[fi]
        for gi in range(nf):
            tri_g = tris[gi]
            n_g = normals[gi]
            # Компланарность (включая self): все вершины tri_f в плоскости tri_g ⇒ ядро ≡ 0.
            dist_plane = float(np.max(np.abs((tri_f - tri_g[0]) @ n_g)))
            if dist_plane < 1.0e-9 * max(diams[fi], diams[gi], 1.0):
                continue

            relation = face_pair_relation(surface_mesh, fi, gi, near_factor=config.near_factor)

            if use_ss and relation in (FacePairRelation.SHARED_EDGE, FacePairRelation.SHARED_VERTEX):
                # Касающаяся непланарная пара → квадратура Заутера–Шваба; trial-базис
                # λ_m возвращается в ПЕРЕУПОРЯДОЧЕННОМ порядке вершин → раскладка по new_g.
                if relation == FacePairRelation.SHARED_EDGE:
                    sg = shared_vertices(surface_mesh, fi, gi)
                    tx, _gx = _ssq.reorder_common_edge(tri_f, surface_mesh.faces[fi], sg)
                    ty, new_g = _ssq.reorder_common_edge(tri_g, surface_mesh.faces[gi], sg)
                    vec3 = _ssq.double_layer_pair_p1(tx, ty, n_g, _ssq.EDGE, config.ss_order)
                else:
                    sgv = shared_vertices(surface_mesh, fi, gi)[0]
                    tx, _gx = _ssq.reorder_common_vertex(tri_f, surface_mesh.faces[fi], sgv)
                    ty, new_g = _ssq.reorder_common_vertex(tri_g, surface_mesh.faces[gi], sgv)
                    vec3 = _ssq.double_layer_pair_p1(tx, ty, n_g, _ssq.VERTEX, config.ss_order)
                for loc in range(3):
                    K[test_dof, phi_space.vertex_to_dof[int(new_g[loc])]] += vec3[loc]
            else:
                # Дальние/near непланарные пары (ядро ограничено) — регулярное
                # тензорное правило высокого порядка БЕЗ подразбиения (regular_order>0),
                # иначе прежний адаптивный путь.
                v0_orig, p_orig = bary[gi]
                if config.regular_order > 0:
                    vec3 = _double_layer_pair_regular_vec3(
                        tri_f, tri_g, n_g, v0_orig, p_orig, reg_pts, reg_wts
                    )
                else:
                    vec3 = _double_layer_pair_adaptive_vec3(
                        tri_f, tri_g, n_g, v0_orig, p_orig, qpts, qw, config.near_factor, config.max_depth
                    )
                phi_dofs = face_phi_dofs[gi]
                for loc in range(3):
                    K[test_dof, phi_dofs[loc]] += vec3[loc]

    return K

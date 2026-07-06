from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.bem.adaptive_single_layer import AdaptiveIntegrationConfig, assemble_single_layer_p0p0_full
from magcore.bem.double_layer import assemble_double_layer_galerkin
from magcore.bem.hypersingular import assemble_hypersingular_maue
from magcore.femcore.assembly import assemble_mixed_coulomb_blocks
from magcore.femcore.basis_nedelec import physical_nedelec_curl
from magcore.femcore.reference_tetra import AffineTetraMap
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.hybrid.interface import CouplingInterface


def assemble_coupling_block(
    interface: CouplingInterface,
    vector_space: NedelecP1Space,
) -> np.ndarray:
    """
    Интерфейсный блок B_FΓ из (FEM)-уравнения связанной задачи (docs/math/coupling.md §3):

        B_FΓ[α, m] = ∮_Γ λ_m^Γ (curl w_α · n) dS,

    связывает A-дофы Неделека (строки α) со следом ψ (P1 по вершинам Γ, столбцы m).

    Поскольку curl w_α постоянен на ячейке (Неделек 1-го рода), а n и P1-базис λ_m
    линейны на плоской грани (∫_f λ_m dS = A_f/3), вклад грани f (владелец c,
    внешняя нормаль n_f, площадь A_f) точен:

        B_FΓ[α, m] += s_α (curl w_local_α · n_f) · (A_f / 3),  α ∈ рёбра(c), m ∈ вершины(f).

    Возвращает плотную матрицу формы (vector_space.ndofs, interface.n_phi_dofs).
    """
    if vector_space.mesh is not interface.tetra_mesh:
        raise ValueError("vector_space must be built on interface.tetra_mesh.")

    n_a = vector_space.ndofs
    n_phi = interface.n_phi_dofs
    B = np.zeros((n_a, n_phi), dtype=float)

    surface = interface.surface_mesh
    vertex_to_dof = interface.phi_space.vertex_to_dof

    for f in range(interface.n_faces):
        c = int(interface.face_to_cell[f])
        normal = interface.outward_normals[f]
        third_area = float(interface.face_areas[f]) / 3.0

        amap = AffineTetraMap(interface.tetra_mesh.cell_vertices(c))
        edge_dofs = vector_space.cell_dof_indices(c)
        edge_signs = vector_space.cell_dof_signs(c)
        phi_dofs = [int(vertex_to_dof[int(v)]) for v in surface.faces[f]]

        for i in range(6):
            curl_n = float(np.dot(physical_nedelec_curl(amap, i), normal))
            contrib = float(edge_signs[i]) * curl_n * third_area
            row = int(edge_dofs[i])
            for m in phi_dofs:
                B[row, m] += contrib

    return B


def assemble_interface_mass(interface: CouplingInterface) -> np.ndarray:
    """
    Интерфейсная масса/дуальность M_PG = ⟨ψ, μ⟩_Γ (P1-вершины × P0-грани), блок «½I»
    комбинаций Кальдерона (docs/math/coupling_block_system.md §4):

        M_PG[m, f] = ∫_{T_f} λ_m^{P1} dS = A_f / 3   (m ∈ вершины f), иначе 0.

    Возвращает (n_phi_dofs × n_flux_dofs). Проверки: Σ_m M_PG[m,f] = A_f; Σ = площадь Γ.
    """
    nv = interface.n_phi_dofs
    nf = interface.n_faces
    surface = interface.surface_mesh
    vertex_to_dof = interface.phi_space.vertex_to_dof

    mass = np.zeros((nv, nf), dtype=float)
    for f in range(nf):
        third = float(interface.face_areas[f]) / 3.0
        for v in surface.faces[f]:
            mass[int(vertex_to_dof[int(v)]), f] += third
    return mass


@dataclass(frozen=True)
class ExteriorOperators:
    """Galerkin-операторы внешней BEM-задачи на Γ + оператор Стеклова–Пуанкаре."""

    V: np.ndarray       # single-layer P0×P0 (nf×nf), SPD
    K: np.ndarray       # double-layer P0×P1 (nf×nv); K' = Kᵀ
    W: np.ndarray       # hypersingular P1×P1 (nv×nv), ⪰0
    M_pg: np.ndarray    # интерфейсная масса P1×P0 (nv×nf)
    S_ext: np.ndarray   # экстерьерный Стеклов–Пуанкаре (nv×nv)
    mu0: float
    k_sign: float


def assemble_exterior_steklov_poincare(
    interface: CouplingInterface,
    config: AdaptiveIntegrationConfig | None = None,
    mu0: float = 1.0,
    k_sign: float = -1.0,
) -> ExteriorOperators:
    """
    Экстерьерный оператор Стеклова–Пуанкаре (DtN) внешней области, симметричная форма
    Costabel (docs/math/coupling_block_system.md §6):

        S_ext = μ₀ [ W + (½I + k_sign·K')·V⁻¹·(½I + k_sign·K) ],

    дискретно (Galerkin): A_int = ½ M_PGᵀ + k_sign·K  (nf×nv),  A_ext = A_intᵀ,

        S_ext = μ₀ ( W + A_intᵀ V⁻¹ A_int ),   симметричный, ⪰0.

    **Знак закреплён физикой/тестом:** внешняя DtN положительно ОПРЕДЕЛЕНА (нет ядра
    констант) — постоянный след даёт ненулевой убывающий внешний поток. При `k_sign=-1`
    (т.е. `½I−K`) `A_int·1 = ½·areas − (−½·areas) = areas ≠ 0` ⇒ `S_ext·1 ≠ 0` ⇒ SPD.
    При `k_sign=+1` `A_int·1 = 0` ⇒ паразитное ядро констант (это поведение ВНУТРЕННЕЙ
    DtN) ⇒ неверно. Поэтому **k_sign=−1** — правильная экстерьерная ветвь Кальдерона.
    """
    surface = interface.surface_mesh
    v_mat, _ = assemble_single_layer_p0p0_full(surface, interface.flux_space.dof_to_face, config)
    k_mat = assemble_double_layer_galerkin(
        surface, interface.outward_normals, interface.phi_space, interface.flux_space, config
    )
    w_mat = assemble_hypersingular_maue(
        surface, interface.outward_normals, interface.phi_space, interface.flux_space, config,
        s_matrix=v_mat,  # переиспользуем уже собранный single-layer V (= S для Мауэ)
    )
    m_pg = assemble_interface_mass(interface)

    a_int = 0.5 * m_pg.T + k_sign * k_mat  # (nf×nv): (½I + k_sign·K), тест P0
    s_ext = mu0 * (w_mat + a_int.T @ np.linalg.solve(v_mat, a_int))
    s_ext = 0.5 * (s_ext + s_ext.T)  # симметризация (гасит ошибку округления V⁻¹)

    return ExteriorOperators(V=v_mat, K=k_mat, W=w_mat, M_pg=m_pg, S_ext=s_ext, mu0=mu0, k_sign=k_sign)


@dataclass(frozen=True)
class CoupledBlockSystem:
    """
    Симметричная связанная блок-система FEM/BEM (новизна 1), неизвестные [a, p, ψ, λ]:

        ⎡ K     G     B_FΓ    0   ⎤ ⎡a⎤   ⎡ f ⎤
        ⎢ Gᵀ    0     0       0   ⎥ ⎢p⎥ = ⎢ 0 ⎥
        ⎢ B_FΓᵀ 0     μ₀W    −C_ΓB⎥ ⎢ψ⎥   ⎢ 0 ⎥
        ⎣ 0     0    −C_ΓBᵀ  −V_BB⎦ ⎣λ⎦   ⎣ 0 ⎦

    `[[K,G],[Gᵀ,0]]` — внутренняя седловая A-Coulomb (formulation_bounded §7);
    `B_FΓ=⟨ψ,curl v·n⟩_Γ`; `C_ΓB=½M_PG−Kᵀ` (знак закреплён, см. coupling_block_system §6);
    `V_BB`=single-layer, `μ₀W`=гиперсингулярный. Симметрична по построению.
    """

    matrix: np.ndarray
    rhs: np.ndarray
    n_a: int
    n_p: int
    n_psi: int
    n_lam: int

    @property
    def off_p(self) -> int:
        return self.n_a

    @property
    def off_psi(self) -> int:
        return self.n_a + self.n_p

    @property
    def off_lam(self) -> int:
        return self.n_a + self.n_p + self.n_psi

    def split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Разбить вектор решения на (a, p, ψ, λ)."""
        return (
            x[: self.off_p],
            x[self.off_p : self.off_psi],
            x[self.off_psi : self.off_lam],
            x[self.off_lam :],
        )


@dataclass(frozen=True)
class CoupledExteriorCache:
    """
    Кэш ЛИНЕЙНЫХ, ν/намагниченность-НЕЗАВИСИМЫХ блоков связанной системы: интерфейсный
    `B_FΓ` и внешние BEM-операторы `ExteriorOperators` (V/K/W/M_PG/S_ext). Зависят только
    от (interface, vector_space, mu0, config) ⇒ в нелинейном Picard собираются ОДИН РАЗ и
    переиспользуются на всех итерациях. Это ТОЧНАЯ оптимизация (результат бит-в-бит тот же,
    проверяется тестом), а не приближение: между итерациями меняются лишь K(ν) и RHS-источник.
    """

    coupling_block: np.ndarray      # B_FΓ (n_a × n_psi)
    ext: ExteriorOperators


def prepare_coupled_exterior(
    interface: CouplingInterface,
    vector_space: NedelecP1Space,
    *,
    mu0: float = 1.0,
    config: AdaptiveIntegrationConfig | None = None,
    k_sign: float = -1.0,
) -> CoupledExteriorCache:
    """Собрать кэшируемые линейные блоки (дорогой BEM-экстерьер + B_FΓ) один раз."""
    return CoupledExteriorCache(
        coupling_block=assemble_coupling_block(interface, vector_space),
        ext=assemble_exterior_steklov_poincare(interface, config=config, mu0=mu0, k_sign=k_sign),
    )


def assemble_coupled_block_system(
    interface: CouplingInterface,
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
    nu,
    j_fn,
    extra_vector_rhs: np.ndarray | None = None,
    applied_field_h0: np.ndarray | None = None,
    mu0: float = 1.0,
    config: AdaptiveIntegrationConfig | None = None,
    exterior: CoupledExteriorCache | None = None,
    curl_quadrature_order: int = 1,
    coupling_quadrature_order: int = 2,
    rhs_quadrature_order: int = 3,
) -> CoupledBlockSystem:
    """
    Собрать симметричную связанную блок-систему (см. CoupledBlockSystem).

    `nu`, `j_fn` — внутренние материал/ток (контраст проницаемости живёт в `nu`).
    `nu` — СКАЛЯР (однородный внутренний материал) ИЛИ массив `(n_cells,)` поячеечно:
    последнее позволяет задать НЕСКОЛЬКО материальных областей в одной задаче
    (воздух ν=1/μ₀ + сталь ν_steel + магнит ν_rec=1/μ_rec) — основа для совместного
    расчёта «магнит ↔ железо». Поячеечная ν протекает через `assemble_mixed_coulomb_blocks`
    (`_nu_per_cell`); внешний BEM-блок зависит только от `mu0`, не от `nu`.
    `extra_vector_rhs` (опц., длины n_a) — дополнительный объёмный источник по A-дофам
    (напр., намагниченность `(νB_r,curl v)` из `assemble_magnetization_rhs`, ненулевая
    только в ячейках магнита).

    **μ₀ и физическая корректность:** при `mu0=1` (безразмерная постановка; для
    бенчмарка сферы контраст μ задаётся в `nu`) дополнение Шура на ψ при исключении λ
    равно `W + C_ΓB V⁻¹ C_ΓBᵀ = S_ext` (знак-закреплённый внешний DtN) — проверяется
    тестом. Размерное `μ₀≠1` требует масштабированного множителя (вывод отложен;
    не нужен для верификации сферой через контраст в `nu`).
    """
    mesh = interface.tetra_mesh
    if vector_space.mesh is not mesh or scalar_space.mesh is not mesh:
        raise ValueError("vector_space/scalar_space must be built on interface.tetra_mesh.")

    k_mat, g_mat, f_vec = assemble_mixed_coulomb_blocks(
        mesh=mesh,
        vector_space=vector_space,
        scalar_space=scalar_space,
        nu=nu,
        J_fn=j_fn,
        curl_quadrature_order=curl_quadrature_order,
        coupling_quadrature_order=coupling_quadrature_order,
        rhs_quadrature_order=rhs_quadrature_order,
    )
    if extra_vector_rhs is not None:
        extra = np.asarray(extra_vector_rhs, dtype=float)
        if extra.shape != (vector_space.ndofs,):
            raise ValueError("extra_vector_rhs must have shape (vector_space.ndofs,).")
        f_vec = f_vec + extra

    if exterior is None:
        b_coupling = assemble_coupling_block(interface, vector_space)  # (n_a × n_psi)
        ext = assemble_exterior_steklov_poincare(interface, config=config, mu0=mu0)
    else:
        # Переиспользуем кэш линейных блоков (нелинейный Picard); mu0 ДОЛЖЕН совпадать,
        # т.к. S_ext/W внутри `ext` уже масштабированы на mu0 кэша.
        if float(exterior.ext.mu0) != float(mu0):
            raise ValueError("exterior cache built with different mu0 than requested.")
        if exterior.coupling_block.shape != (vector_space.ndofs, interface.n_phi_dofs):
            raise ValueError("exterior.coupling_block has wrong shape for this interface/space.")
        b_coupling = exterior.coupling_block
        ext = exterior.ext
    c_gb = 0.5 * ext.M_pg - ext.K.T  # C_ΓB = ½M_PG − Kᵀ  (n_psi × n_lam), знак закреплён

    n_a = vector_space.ndofs
    n_p = scalar_space.ndofs
    n_psi = interface.n_phi_dofs
    n_lam = interface.n_flux_dofs
    n = n_a + n_p + n_psi + n_lam
    op, opsi, olam = n_a, n_a + n_p, n_a + n_p + n_psi

    m = np.zeros((n, n), dtype=float)
    # внутренняя седловая
    m[:n_a, :n_a] = k_mat
    m[:n_a, op:opsi] = g_mat
    m[op:opsi, :n_a] = g_mat.T
    # связь A↔ψ
    m[:n_a, opsi:olam] = b_coupling
    m[opsi:olam, :n_a] = b_coupling.T
    # Внешний 2-блок входит со ЗНАКОМ МИНУС (отн. изолированного S_ext): −μ₀W, +C_ΓB, +V.
    # Тогда дополнение Шура на ψ = −S_ext, и конденсированная задача даёт
    # K_eff = K + B_FΓ S_ext⁻¹ B_FΓᵀ (внешняя реакция ДОБАВЛЯет жёсткость ⇒
    # размагничивание). Знак выведен из слабой формы (IC1−E2) и ПОДТВЕРЖДЁН бенчмарком
    # намагниченной сферы H_in=−M/3 (энергетическая эвристика «совместной минимизации»
    # вводит в заблуждение — это задача сопряжения/седло, а не joint-min).
    m[opsi:olam, opsi:olam] = -mu0 * ext.W
    m[opsi:olam, olam:] = c_gb
    m[olam:, opsi:olam] = c_gb.T
    m[olam:, olam:] = ext.V

    rhs = np.zeros(n, dtype=float)
    rhs[:n_a] = f_vec

    # Однородное приложенное поле H₀ (открытая область): расщепление ψ=ψ_app+ψ_scat,
    # ψ_app=−H₀·x (не убывает, не часть BEM). Известные члены → RHS (вывод: coupling_block_system §10доп):
    #   a-строка (IC2):  f_a = −B_FΓ ψ_app^Γ,   ψ_app^Γ[m] = −H₀·x_m;
    #   ψ-строка (IC1):  f_ψ = ⟨H₀·n, ξ⟩ = M_PG·(H₀·n_f).
    if applied_field_h0 is not None:
        h0 = np.asarray(applied_field_h0, dtype=float)
        if h0.shape != (3,):
            raise ValueError("applied_field_h0 must have shape (3,).")
        sv = interface.surface_mesh.vertices
        vert_idx = np.asarray(interface.phi_space.dof_to_vertex, dtype=int)
        psi_app = -(sv[vert_idx] @ h0)               # ψ_app = −H₀·x на вершинах Γ (n_psi,)
        rhs[:n_a] += -(b_coupling @ psi_app)          # f_a = −B_FΓ ψ_app
        g_n = interface.outward_normals @ h0          # H₀·n_f по граням (n_faces,)
        rhs[opsi:olam] += ext.M_pg @ g_n              # f_ψ = M_PG·(H₀·n)

    return CoupledBlockSystem(matrix=m, rhs=rhs, n_a=n_a, n_p=n_p, n_psi=n_psi, n_lam=n_lam)

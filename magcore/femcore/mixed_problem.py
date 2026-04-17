from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from magcore.femcore.assembly import (
    assemble_mixed_coulomb_system,
)
from magcore.femcore.boundary_conditions import (
    apply_zero_mixed_dirichlet_bc,
    find_mixed_boundary_dofs,
)
from magcore.femcore.gauge_diagnostics import (
    GaugeProjectionResult,
    gauge_residual_norm,
    gauge_residual_vector,
    project_to_gradient_subspace,
)
from magcore.femcore.mesh import TetraMesh
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.solver import (
    solve_mixed_coulomb_problem,
    split_mixed_solution,
)
from magcore.femcore.spaces import NedelecP1Space


@dataclass(frozen=True, slots=True)
class MixedCoulombProblem:
    """
    Высокоуровневое описание bounded mixed A-p задачи Кулона
    для этапа A1.2.

    Полная дискретная система имеет вид:
        [ K   G ] [a] = [f]
        [ G^T 0 ] [p]   [0]

    где:
    - a — коэффициенты векторного потенциала A_h,
    - p — коэффициенты множителя калибровки p_h.
    """

    mesh: TetraMesh
    vector_space: NedelecP1Space
    scalar_space: LagrangeP1Space
    nu: float
    J_fn: object
    curl_quadrature_order: int = 1
    coupling_quadrature_order: int = 2
    rhs_quadrature_order: int = 2

    def __post_init__(self) -> None:
        if self.vector_space.mesh is not self.mesh:
            raise ValueError("vector_space must be built on the provided mesh.")
        if self.scalar_space.mesh is not self.mesh:
            raise ValueError("scalar_space must be built on the provided mesh.")
        if self.nu <= 0.0:
            raise ValueError("nu must be positive.")
        if not callable(self.J_fn):
            raise ValueError("J_fn must be callable.")
        if self.curl_quadrature_order not in (1, 2):
            raise ValueError("curl_quadrature_order must be 1 or 2.")
        if self.coupling_quadrature_order not in (1, 2):
            raise ValueError("coupling_quadrature_order must be 1 or 2.")
        if self.rhs_quadrature_order not in (1, 2):
            raise ValueError("rhs_quadrature_order must be 1 or 2.")

    @classmethod
    def from_mesh(
        cls,
        mesh: TetraMesh,
        nu: float,
        J_fn,
        curl_quadrature_order: int = 1,
        coupling_quadrature_order: int = 2,
        rhs_quadrature_order: int = 2,
    ) -> "MixedCoulombProblem":
        """
        Удобный конструктор: автоматически создаёт H(curl)- и P1-пространства.
        """
        vector_space = NedelecP1Space.from_mesh(mesh)
        scalar_space = LagrangeP1Space(mesh)

        return cls(
            mesh=mesh,
            vector_space=vector_space,
            scalar_space=scalar_space,
            nu=nu,
            J_fn=J_fn,
            curl_quadrature_order=curl_quadrature_order,
            coupling_quadrature_order=coupling_quadrature_order,
            rhs_quadrature_order=rhs_quadrature_order,
        )

    @property
    def n_vector_dofs(self) -> int:
        return self.vector_space.ndofs

    @property
    def n_scalar_dofs(self) -> int:
        return self.scalar_space.ndofs

    @property
    def n_total_dofs(self) -> int:
        return self.n_vector_dofs + self.n_scalar_dofs

    def boundary_dofs(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """
        Вернуть граничные DOF для:
        - векторного блока A_h,
        - скалярного блока p_h.
        """
        return find_mixed_boundary_dofs(
            vector_space=self.vector_space,
            scalar_space=self.scalar_space,
        )

    def assemble_system(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Собрать полную mixed-систему без наложенных граничных условий.
        """
        return assemble_mixed_coulomb_system(
            mesh=self.mesh,
            vector_space=self.vector_space,
            scalar_space=self.scalar_space,
            nu=self.nu,
            J_fn=self.J_fn,
            curl_quadrature_order=self.curl_quadrature_order,
            coupling_quadrature_order=self.coupling_quadrature_order,
            rhs_quadrature_order=self.rhs_quadrature_order,
        )

    def apply_boundary_conditions(
        self,
        system_matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Наложить граничные условия:
        - n × A = 0,
        - p = 0.
        """
        vector_bnd, scalar_bnd = self.boundary_dofs()

        return apply_zero_mixed_dirichlet_bc(
            A=system_matrix,
            b=rhs,
            vector_dofs=vector_bnd,
            scalar_dofs=scalar_bnd,
            n_vector_dofs=self.n_vector_dofs,
        )

    def solve(
        self,
        *,
        compute_gauge_projection: bool = True,
        store_full_system: bool = False,
    ) -> "MixedCoulombSolution":
        """
        Полный цикл решения задачи A1.2:
        1. сборка,
        2. наложение ГУ,
        3. решение,
        4. диагностика калибровки.
        """
        system_matrix, rhs = self.assemble_system()
        system_matrix_bc, rhs_bc = self.apply_boundary_conditions(system_matrix, rhs)

        x = solve_mixed_coulomb_problem(system_matrix_bc, rhs_bc)
        a, p = split_mixed_solution(x, self.n_vector_dofs)

        G = system_matrix[: self.n_vector_dofs, self.n_vector_dofs :]

        linear_residual = system_matrix_bc @ x - rhs_bc
        linear_residual_norm = float(np.linalg.norm(linear_residual))

        gauge_vec = gauge_residual_vector(G, a)
        gauge_norm = gauge_residual_norm(G, a)

        gauge_projection: GaugeProjectionResult | None
        if compute_gauge_projection:
            gauge_projection = project_to_gradient_subspace(
                mesh=self.mesh,
                vector_space=self.vector_space,
                scalar_space=self.scalar_space,
                a=a,
            )
        else:
            gauge_projection = None

        if store_full_system:
            stored_system_matrix = system_matrix
            stored_rhs = rhs
            stored_system_matrix_bc = system_matrix_bc
            stored_rhs_bc = rhs_bc
        else:
            stored_system_matrix = None
            stored_rhs = None
            stored_system_matrix_bc = None
            stored_rhs_bc = None

        return MixedCoulombSolution(
            problem=self,
            x=x,
            a=a,
            p=p,
            linear_residual=linear_residual,
            linear_residual_norm=linear_residual_norm,
            gauge_residual=gauge_vec,
            gauge_residual_norm=gauge_norm,
            gauge_projection=gauge_projection,
            system_matrix=stored_system_matrix,
            rhs=stored_rhs,
            system_matrix_bc=stored_system_matrix_bc,
            rhs_bc=stored_rhs_bc,
        )


@dataclass(frozen=True, slots=True)
class MixedCoulombSolution:
    """
    Результат решения mixed A-p задачи Кулона.
    """

    problem: MixedCoulombProblem
    x: np.ndarray
    a: np.ndarray
    p: np.ndarray
    linear_residual: np.ndarray
    linear_residual_norm: float
    gauge_residual: np.ndarray
    gauge_residual_norm: float
    gauge_projection: GaugeProjectionResult | None
    system_matrix: np.ndarray | None = None
    rhs: np.ndarray | None = None
    system_matrix_bc: np.ndarray | None = None
    rhs_bc: np.ndarray | None = None

    @property
    def eta_parallel(self) -> float | None:
        """
        Относительная продольная составляющая:
            eta_parallel = ||A_parallel|| / ||A||
        """
        if self.gauge_projection is None:
            return None
        return float(self.gauge_projection.eta_parallel)

    @property
    def norm_total(self) -> float | None:
        if self.gauge_projection is None:
            return None
        return float(self.gauge_projection.norm_total)

    @property
    def norm_parallel(self) -> float | None:
        if self.gauge_projection is None:
            return None
        return float(self.gauge_projection.norm_parallel)

    @property
    def norm_perp(self) -> float | None:
        if self.gauge_projection is None:
            return None
        return float(self.gauge_projection.norm_perp)

    def summary_dict(self) -> dict[str, float | int | None]:
        """
        Краткая сводка по решению в удобном для логирования виде.
        """
        return {
            "n_vector_dofs": self.problem.n_vector_dofs,
            "n_scalar_dofs": self.problem.n_scalar_dofs,
            "n_total_dofs": self.problem.n_total_dofs,
            "linear_residual_norm": self.linear_residual_norm,
            "gauge_residual_norm": self.gauge_residual_norm,
            "eta_parallel": self.eta_parallel,
            "norm_total": self.norm_total,
            "norm_parallel": self.norm_parallel,
            "norm_perp": self.norm_perp,
        }


def solve_mixed_coulomb_baseline(
    mesh: TetraMesh,
    nu: float,
    J_fn,
    *,
    curl_quadrature_order: int = 1,
    coupling_quadrature_order: int = 2,
    rhs_quadrature_order: int = 2,
    compute_gauge_projection: bool = True,
    store_full_system: bool = False,
) -> MixedCoulombSolution:
    """
    Функциональный интерфейс поверх MixedCoulombProblem.

    Это удобная точка входа для:
    - тестов,
    - скриптов,
    - будущих verification-case сценариев.
    """
    problem = MixedCoulombProblem.from_mesh(
        mesh=mesh,
        nu=nu,
        J_fn=J_fn,
        curl_quadrature_order=curl_quadrature_order,
        coupling_quadrature_order=coupling_quadrature_order,
        rhs_quadrature_order=rhs_quadrature_order,
    )

    return problem.solve(
        compute_gauge_projection=compute_gauge_projection,
        store_full_system=store_full_system,
    )
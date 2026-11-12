"""The CPU Riccati ADMM (admm_fused_riccati_cpu) against the JAX loop in fixed-iteration mode.

Both run the same ADMM iterations (fixed rho, no checks inside the loop), so solutions and
gradients through the differentiable solve agree up to floating-point rounding.
"""
import gc

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers.problem_fixtures import make_linear_params
from turbompc.problems.obstacle_avoidance import OptimalControlProblemObstacle
from turbompc.problems.optimal_control_problem import OptimalControlProblem
from turbompc.solvers.turbompc_solver import (
    BackwardBackend,
    ForwardBackend,
    TurboMPCSolver,
)
from turbompc.utils.load_params import load_solver_params

try:
    import turbompc.solvers.admm.admm_riccati_cpu_backend  # noqa: F401
except (FileNotFoundError, OSError, ImportError) as error:  # pragma: no cover
    pytest.skip(f"CPU Riccati ADMM library not built: {error}", allow_module_level=True)


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    yield
    jax.clear_caches()
    gc.collect()


def _solver_params(admm_iterations, threads=1, sqp_iterations=5):
    params = dict(load_solver_params("turbompc.yaml"))
    params["num_sqp_iteration_max"] = sqp_iterations
    params["admm"] = dict(params["admm"])
    params["admm"].update(max_iter=admm_iterations, fixed_iterations=True, num_threads=threads)
    return params


def _pair(problem, params, backward=BackwardBackend.DIRECT_JAX_DENSE):
    reference = TurboMPCSolver(
        problem,
        params,
        forward_backend=ForwardBackend.ADMM_JAX_LOOP_JAX_DENSE,
        backward_backend=backward,
    )
    riccati = TurboMPCSolver(
        problem,
        params,
        forward_backend="admm_fused_riccati_cpu",
        backward_backend=backward,
    )
    return reference, riccati


def _relative(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))


@pytest.mark.parametrize(
    "implicit, control_rate, initial_control, threads",
    [
        (False, 0.0, False, 1),  # controls eliminated first, state chain
        (False, 0.0, True, 3),  # constrained initial control, threaded
        (False, 0.5, False, 2),  # cost cross terms: chain over whole nodes
        (True, 0.0, False, 1),  # implicit integrator: A_plus has control columns
    ],
)
def test_matches_fixed_iteration_jax_loop(implicit, control_rate, initial_control, threads):
    dynamics, problem_params = make_linear_params(
        12, implicit=implicit, bounded=True, initial_control=initial_control
    )
    problem_params = dict(problem_params)
    problem_params["weights_penalization_control_rate"] = control_rate * jnp.ones(
        (dynamics.num_controls,)
    )
    # Tight bounds so that the projection is active.
    problem_params["control_min_bounds"] = -0.3 * jnp.ones((dynamics.num_controls,))
    problem_params["control_max_bounds"] = 0.3 * jnp.ones((dynamics.num_controls,))
    problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)
    reference, riccati = _pair(problem, _solver_params(40, threads))
    guess = reference.initial_guess(problem_params)
    expected = jax.jit(reference.solve)(guess, problem_params)
    actual = jax.jit(riccati.solve)(guess, problem_params)
    assert int(actual.stats.qp_stats.num_iter[0]) == 40
    for name in ("states", "controls"):
        error = _relative(getattr(actual, name), getattr(expected, name))
        assert error < 1e-9, (name, error)
    for name in ("y_g", "y_f_0", "y_f_dyn", "z_g"):
        error = _relative(getattr(actual.admm_state, name), getattr(expected.admm_state, name))
        assert error < 1e-8, (name, error)
    np.testing.assert_array_equal(
        np.asarray(actual.stats.qp_stats.status), np.asarray(expected.stats.qp_stats.status)
    )


def test_batched_solve_matches_unbatched():
    dynamics, problem_params = make_linear_params(8, implicit=False, bounded=True)
    problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)
    _, riccati = _pair(problem, _solver_params(25, 2, sqp_iterations=2))
    guess = riccati.initial_guess(problem_params)
    states0 = jnp.stack([problem_params["initial_state"] * s for s in (1.0, 0.5, -0.7)])

    def solve(x0):
        params = dict(problem_params, initial_state=x0)
        return riccati.solve(guess, params).states

    batched = jax.jit(jax.vmap(solve))(states0)
    for index in range(states0.shape[0]):
        single = jax.jit(solve)(states0[index])
        assert _relative(batched[index], single) < 1e-12


def _obstacle_problem():
    horizon, dt = 5, 0.1
    start = jnp.array([0.0, 0.0, 0.0, 0.0])
    goal = jnp.array([2.0, 1.5, 0.0, 0.0])
    dynamics, problem_params = make_linear_params(
        horizon, implicit=False, bounded=False, initial_state=start
    )
    problem_params = dict(problem_params)
    problem_params.update(
        discretization_resolution=dt,
        initial_guess_final_state=goal,
        reference_state_trajectory=jnp.tile(goal, (horizon + 1, 1)),
        weights_penalization_reference_state_trajectory=jnp.array([5.0, 5.0, 0.0, 0.0]),
        weights_penalization_control_squared=jnp.array([0.1, 0.1]),
        weights_penalization_final_state=jnp.zeros((4,)),
        dynamics_state_dot_params={
            "A": jnp.array(
                [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0] * 4, [0.0] * 4]
            ),
            "B": jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            "b": jnp.zeros((4,)),
        },
        obstacles_centers=jnp.tile(jnp.array([1.0, 0.55])[None, None, :], (horizon + 1, 1, 1)),
        obstacles_radii=jnp.array([0.55]),
        obstacles_dimension=2,
        rescale_optimization_variables=False,
    )
    return OptimalControlProblemObstacle(dynamics=dynamics, params=problem_params), problem_params, goal


@pytest.mark.parametrize("backward", [BackwardBackend.DIRECT_JAX_DENSE, BackwardBackend.ADMM_JAX_LOOP_JAX_DENSE])
def test_gradients_match(backward):
    problem, problem_params, goal = _obstacle_problem()
    reference, riccati = _pair(problem, _solver_params(200, 2, sqp_iterations=20), backward)
    keys = ("weights_penalization_reference_state_trajectory", "weights_penalization_control_squared")

    def gradient(solver):
        guess = solver.initial_guess(problem_params)

        def loss(weights):
            solution = solver.solve(guess, problem_params, {**{k: problem_params[k] for k in keys}, **weights})
            return 0.5 * jnp.sum((solution.states[:, :2] - goal[:2]) ** 2) + 5e-3 * jnp.sum(solution.controls**2)

        return jax.jit(jax.grad(loss))({k: problem_params[k] for k in keys})

    expected, actual = gradient(reference), gradient(riccati)
    for key in keys:
        assert jnp.all(jnp.isfinite(actual[key]))
        assert _relative(actual[key], expected[key]) < 1e-6, (key, actual[key], expected[key])

"""Fused CPU ADMM with a Riccati-type factorization (XLA FFI, opt-in).

The whole ADMM loop runs in one native call: the x-update matrix
``P + sigma I + rho G'G + rho_f C'C`` is factored once per QP with a block
Cholesky that eliminates every node's controls first and then runs a chain over
the states (see ``admm_riccati_cpu.cpp``), and each of the ``max_iter``
iterations only substitutes. It computes the same iterates as
``ADMMSolver``'s JAX loop with ``fixed_iterations=True`` (projection handling,
no slack variables, fixed rho), up to floating-point rounding.
"""

from __future__ import annotations

import ctypes

import jax
import jax.numpy as jnp

from turbompc.solvers.admm.admm import (
    ADMMState,
    ADMMStats,
    FusedADMMSolveResult,
)
from turbompc.solvers.ffi_loader import find_ffi_lib
from turbompc.solvers.qp_data import QPData
from turbompc.solvers.status import ADMMStatus

_LIB_NAME = "libadmm_riccati_cpu.so"
_LIB = ctypes.cdll.LoadLibrary(
    str(find_ffi_lib(_LIB_NAME, env_var="TURBOMPC_RICCATI_CPU_LIB"))
)

jax.ffi.register_ffi_target(
    "admm_riccati_cpu", jax.ffi.pycapsule(_LIB.AdmmRiccatiCpu), platform="cpu"
)
jax.ffi.register_ffi_target(
    "admm_riccati_cpu_f64", jax.ffi.pycapsule(_LIB.AdmmRiccatiCpuF64), platform="cpu"
)

# info layout returned by the native call (see kInfoSize in admm_riccati_cpu.cpp).
_PRIMAL, _DUAL, _PRIMAL_NORM, _DUAL_NORM, _FINITE = 0, 1, 2, 3, 4
_INFO_SIZE = 10


def admm_riccati_cpu_solve(
    qp_data: QPData,
    state0: ADMMState,
    *,
    max_iter: int,
    sigma: float,
    rho_f_factor: float,
    alpha: float,
    eps_abs: float,
    eps_rel: float,
    num_threads: int = 1,
    return_info: bool = False,
):
    """Run ``max_iter`` ADMM iterations from ``state0`` in one native call.

    Residuals are evaluated once, after the last iteration, for the reported
    status: SOLVED within (eps_abs, eps_rel), else MAX_ITER, or NONFINITE.
    ``kernel_ns`` is the native time. With ``return_info``, also returns the native
    info vector (residuals, structure, factorization and iteration seconds).
    """
    dtype = qp_data.cost.q.dtype
    if dtype not in (jnp.float32, jnp.float64):
        raise TypeError("admm_riccati_cpu supports float32 and float64 QPs")
    target = "admm_riccati_cpu_f64" if dtype == jnp.float64 else "admm_riccati_cpu"
    cast = lambda a: jnp.asarray(a, dtype=dtype)  # noqa: E731
    cost, eq, ineq = qp_data.cost, qp_data.eq, qp_data.ineq
    T, n = cost.q.shape
    if T < 2:
        raise ValueError("admm_riccati_cpu needs a horizon of at least one step")
    m = ineq.G.shape[1]
    n0 = eq.A0.shape[0]
    nx = eq.A_minus.shape[1]
    shape = lambda *s: jax.ShapeDtypeStruct(s, dtype)  # noqa: E731
    call = jax.ffi.ffi_call(
        target,
        (
            shape(T, n),
            shape(T, m),
            shape(n0),
            shape(T - 1, nx),
            shape(T, m),
            shape(_INFO_SIZE),
        ),
        vmap_method="sequential",
    )
    x, y_g, y_f_0, y_f_dyn, z_g, info = call(
        cast(cost.D),
        cast(cost.E),
        cast(cost.q),
        cast(eq.A0),
        cast(eq.A_minus),
        cast(eq.A_plus),
        cast(eq.c0),
        cast(eq.c),
        cast(ineq.G),
        cast(ineq.l),
        cast(ineq.u),
        cast(state0.x_blocks),
        cast(state0.y_g).reshape(T, m),
        cast(state0.y_f_0),
        cast(state0.y_f_dyn),
        cast(state0.z_g).reshape(T, m),
        cast(state0.rho_bar),
        sigma=float(sigma),
        rho_f_factor=float(rho_f_factor),
        alpha=float(alpha),
        max_iter=int(max_iter),
        num_threads=int(num_threads),
    )
    primal_limit = eps_abs + eps_rel * info[_PRIMAL_NORM]
    dual_limit = eps_abs + eps_rel * info[_DUAL_NORM]
    finite = info[_FINITE] > 0.5
    converged = (info[_PRIMAL] <= primal_limit) & (info[_DUAL] <= dual_limit)
    status = jnp.where(
        ~finite,
        jnp.int32(ADMMStatus.NONFINITE),
        jnp.where(
            converged, jnp.int32(ADMMStatus.SOLVED), jnp.int32(ADMMStatus.MAX_ITER)
        ),
    )
    stats = ADMMStats(
        status=status,
        num_iter=jnp.asarray(max_iter, dtype=jnp.int32),
        primal_residual=info[_PRIMAL],
        primal_residual_limit=primal_limit,
        dual_residual=info[_DUAL],
        dual_residual_limit=dual_limit,
    )
    state = ADMMState(
        x_blocks=x,
        y_g=y_g,
        y_f_0=y_f_0,
        y_f_dyn=y_f_dyn,
        z_g=z_g,
        xi_g=jnp.zeros_like(z_g),
        rho_bar=cast(state0.rho_bar),
    )
    result = FusedADMMSolveResult(
        state=state, stats=stats, kernel_ns=1.0e9 * (info[8] + info[9])
    )
    return (result, info) if return_info else result

"""Unconstrained minimizers operating on a differentiable energy.

Both optimizers take the same inputs: a `fun` providing `value(x)`, `grad(x)` and
`hvp(x, v)` (the `semodern.energy.Energy` interface), a flat float64 `x0` of shape
`(3n,)`, and return an `OptResult`.  Only the trust-region solver needs `hvp`, and
it accesses the Hessian exclusively through that product, so no Hessian is ever
assembled.

`lbfgs` is the default: limited-memory BFGS with a strong-Wolfe line search.
`tr_newton_cg` is the high-accuracy option: Steihaug-CG inside a trust region,
which also handles negative curvature by truncating along the negative-curvature
direction rather than failing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch

Tensor = torch.Tensor


class EnergyLike(Protocol):
    """Minimal interface an objective must provide."""

    def value(self, x: Tensor) -> Tensor:
        """Scalar objective at `x`."""

    def grad(self, x: Tensor) -> Tensor:
        """Gradient at `x`, shaped like `x`."""

    def hvp(self, x: Tensor, v: Tensor) -> Tensor:
        """Hessian-vector product at `x`."""


@dataclass
class OptResult:
    """Outcome of an optimization run."""

    x: torch.Tensor
    energy: float
    grad_norm: float
    iterations: int
    converged: bool
    history: list[float] = field(default_factory=list)


def _flat(x) -> torch.Tensor:
    vec = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float64)
    return vec.reshape(-1).detach().to(torch.float64)


def _inf_norm(vec: torch.Tensor) -> float:
    return float(vec.abs().max()) if vec.numel() else 0.0


def strong_wolfe(
    fun: EnergyLike,
    x: torch.Tensor,
    value: float,
    grad: torch.Tensor,
    direction: torch.Tensor,
    *,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_step: float = 1e6,
    max_iter: int = 40,
) -> tuple[float, torch.Tensor, torch.Tensor, float]:
    """Bracketing + zoom line search satisfying the strong Wolfe conditions.

    Returns `(step, x_new, grad_new, value_new)`.  If no step can be found, the
    smallest tried step is returned so the caller can still make progress and
    shrink its own trust region.
    """
    slope = float(grad.dot(direction))
    if slope >= 0.0:
        return 0.0, x, grad, value

    step_prev, value_prev = 0.0, value
    step = 1.0
    for _ in range(max_iter):
        candidate = x + step * direction
        value_candidate = float(fun.value(candidate))
        if not math.isfinite(value_candidate):
            step *= 0.5
            continue
        if value_candidate > value + c1 * step * slope or (step_prev > 0 and value_candidate >= value_prev):
            return _zoom(fun, x, value, grad, direction, slope, step_prev, step, c1, c2)
        grad_candidate = fun.grad(candidate)
        curvature = float(grad_candidate.dot(direction))
        if abs(curvature) <= -c2 * slope:
            return step, candidate, grad_candidate, value_candidate
        if curvature >= 0.0:
            return _zoom(fun, x, value, grad, direction, slope, step, step_prev, c1, c2)
        step_prev, value_prev = step, value_candidate
        step = min(step * 2.0, max_step)
    return step_prev, x + step_prev * direction, fun.grad(x + step_prev * direction), (
        float(fun.value(x + step_prev * direction))
    )


def _zoom(
    fun: EnergyLike,
    x: torch.Tensor,
    value: float,
    grad: torch.Tensor,
    direction: torch.Tensor,
    slope: float,
    low: float,
    high: float,
    c1: float,
    c2: float,
    max_iter: int = 40,
) -> tuple[float, torch.Tensor, torch.Tensor, float]:
    """Zoom between a bracket `[low, high]` to satisfy the strong Wolfe conditions."""
    for _ in range(max_iter):
        step = 0.5 * (low + high)
        if not math.isfinite(step) or step <= 0.0:
            break
        candidate = x + step * direction
        value_candidate = float(fun.value(candidate))
        if not math.isfinite(value_candidate) or value_candidate > value + c1 * step * slope:
            high = step
            continue
        grad_candidate = fun.grad(candidate)
        curvature = float(grad_candidate.dot(direction))
        if abs(curvature) <= -c2 * slope:
            return step, candidate, grad_candidate, value_candidate
        if curvature * (high - low) >= 0.0:
            high = low
        low = step
    step = max(low, 1e-16)
    candidate = x + step * direction
    return step, candidate, fun.grad(candidate), float(fun.value(candidate))


def lbfgs(
    fun: EnergyLike,
    x0,
    *,
    max_iter: int = 500,
    memory: int = 10,
    tol: float = 1e-8,
    callback: Callable[[int, float, float], None] | None = None,
) -> OptResult:
    """Minimize `fun` with L-BFGS and a strong-Wolfe line search.

    Stopping: `||grad||_inf < tol`, or no decrease is achievable along the computed
    direction.  `callback(iteration, energy, grad_norm)` is called every iteration.
    """
    x = _flat(x0)
    value = float(fun.value(x))
    grad = fun.grad(x)
    history = [value]
    callbacks: list[float] = []
    converged = _inf_norm(grad) < tol
    steps: list[torch.Tensor] = []
    curvatures: list[torch.Tensor] = []
    iteration = 0

    while iteration < max_iter and not converged:
        direction = -grad.clone()
        alphas: list[float] = []
        for step, curvature in zip(reversed(steps), reversed(curvatures)):
            alpha = float(curvature.dot(direction)) / float(curvature.dot(step))
            alphas.append(alpha)
            direction = direction - alpha * curvature
        scale = 1.0
        if steps and curvatures:
            scale = float(steps[-1].dot(curvatures[-1])) / float(curvatures[-1].dot(curvatures[-1]))
        direction = scale * direction
        for step, curvature, alpha in zip(steps, curvatures, reversed(alphas)):
            beta = float(curvature.dot(direction)) / float(curvature.dot(step))
            direction = direction + (alpha - beta) * step

        if float(direction.dot(grad)) >= 0.0:
            direction = -grad.clone()
            steps.clear()
            curvatures.clear()

        step_size, x_new, grad_new, value_new = strong_wolfe(fun, x, value, grad, direction)
        if step_size <= 0.0 or value_new > value:
            break

        displacement = x_new - x
        new_grad = grad_new - grad
        curvature = float(new_grad.dot(displacement))
        if curvature > 1e-12 * float(displacement.norm()) * float(new_grad.norm()):
            steps.append(displacement)
            curvatures.append(new_grad)
            if len(steps) > memory:
                steps.pop(0)
                curvatures.pop(0)

        x, grad, value = x_new, grad_new, value_new
        iteration += 1
        history.append(value)
        callbacks.append(_inf_norm(grad))
        if callback is not None:
            callback(iteration, value, _inf_norm(grad))
        converged = _inf_norm(grad) < tol

    return OptResult(
        x=x,
        energy=value,
        grad_norm=_inf_norm(grad),
        iterations=iteration,
        converged=converged,
        history=history,
    )


def steihaug_cg(
    fun: EnergyLike,
    x: torch.Tensor,
    grad: torch.Tensor,
    radius: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[torch.Tensor, bool]:
    """Steihaug-Toint truncated CG solve of `H p = -grad` inside `||p|| <= radius`.

    Returns `(p, hit_boundary)`.  Negative curvature and exceeding the radius both
    terminate early along the current direction, so the result is a descent
    direction even when `H` is indefinite.
    """
    p = torch.zeros_like(grad)
    residual = grad.clone()
    direction = -residual
    residual_norm = float(residual.norm())
    if residual_norm == 0.0:
        return p, False
    for _ in range(max_iter):
        hessian_direction = fun.hvp(x, direction)
        curvature = float(direction.dot(hessian_direction))
        if curvature <= 0.0:
            return _to_boundary(p, direction, radius), True
        alpha = residual_norm**2 / curvature
        p_next = p + alpha * direction
        if float(p_next.norm()) >= radius:
            return _to_boundary(p, direction, radius), True
        p = p_next
        residual = residual + alpha * hessian_direction
        residual_norm_next = float(residual.norm())
        if residual_norm_next <= tol * max(1.0, float(grad.norm())):
            return p, False
        direction = -residual + (residual_norm_next**2 / residual_norm**2) * direction
        residual_norm = residual_norm_next
    return p, False


def _to_boundary(p: torch.Tensor, direction: torch.Tensor, radius: float) -> torch.Tensor:
    """Scale `p + tau*direction` to the trust-region boundary."""
    a = float(direction.dot(direction))
    b = 2.0 * float(p.dot(direction))
    c = float(p.dot(p)) - radius * radius
    if a <= 0.0:
        return p
    discriminant = max(b * b - 4.0 * a * c, 0.0)
    tau = (-b + math.sqrt(discriminant)) / (2.0 * a)
    return p + tau * direction


def tr_newton_cg(
    fun: EnergyLike,
    x0,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
    max_radius: float = 0.1,
    callback: Callable[[int, float, float], None] | None = None,
) -> OptResult:
    """Minimize `fun` by trust-region Newton-CG (Hessian accessed only via `hvp`).

    The radius starts at `max_radius`, doubles after a successful step and quarters
    after a rejected one.  A step is accepted when the ratio of actual to
    predicted decrease exceeds `1e-4`; `||grad||_inf < tol` stops the iteration.
    """
    x = _flat(x0)
    value = float(fun.value(x))
    grad = fun.grad(x)
    history = [value]
    radius = float(max_radius)
    converged = _inf_norm(grad) < tol
    iteration = 0
    accepted = 0

    while iteration < max_iter and not converged:
        direction, hit_boundary = steihaug_cg(fun, x, grad, radius)
        if float(grad.dot(direction)) >= 0.0:
            direction = -grad / max(_inf_norm(grad), 1e-300) * min(radius, _inf_norm(grad))
            hit_boundary = True
        slope = float(grad.dot(direction))
        predicted = -(slope + 0.5 * float(direction.dot(fun.hvp(x, direction))))
        x_new = x + direction
        value_new = float(fun.value(x_new))
        actual = value - value_new
        ratio = actual / predicted if predicted > 0.0 else -math.inf
        iteration += 1

        if math.isfinite(value_new) and actual > 0.0 and ratio > 1e-4:
            x, value = x_new, value_new
            grad = fun.grad(x)
            accepted += 1
            history.append(value)
            if callback is not None:
                callback(iteration, value, _inf_norm(grad))
            converged = _inf_norm(grad) < tol
            if hit_boundary or ratio > 0.75:
                radius = min(2.0 * radius, max_radius * 1e6)
        else:
            radius *= 0.25
            if callback is not None:
                callback(iteration, value, _inf_norm(grad))
            if radius < 1e-14 * max(1.0, float(x.norm())):
                break

    return OptResult(
        x=x,
        energy=value,
        grad_norm=_inf_norm(grad),
        iterations=iteration,
        converged=converged,
        history=history,
    )

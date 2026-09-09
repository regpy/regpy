"""Utilities for plotting convergence histories stored by RegPy stop rules."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from regpy.stoprules import (
    CountIterations,
    DualityGapStopping,
    ErrorMonitor,
    OptimalityCondStopping,
    StopRule,
)

if TYPE_CHECKING:
    from regpy.solvers import Setting

__all__ = ["convergence_plots", "GeometricSequence", "powers_of_two", "natural_numbers", "squares"]

def powers_of_two() -> Iterator[int]:
    """Yield ``1, 2, 4, 8, ...`` indefinitely."""
    n = 0
    while True:
        yield 2**n
        n += 1

def natural_numbers() -> Iterator[int]:
    """Yield ``1, 2, 3, ...`` indefinitely."""
    n = 1
    while True:
        yield n
        n += 1

def squares() -> Iterator[int]:
    """Yield ``0, 1, 4, 9, ...`` indefinitely."""
    n = 0
    while True:
        yield n**2
        n += 1

class GeometricSequence:
    r"""Iterator generating a geometric sequence
    
    Parameters
    ----------
    alpha0 : float
        :math:`\alpha_0` the initial regularization parameter 
    q : float
        Rate of the geometric sequence

    Notes
    ----- 
    Sequence defined recursively by
    
    .. math::
        \alpha_0 &= \alpha_0 \\
        \alpha_{n+1} &= q*\alpha_n
    """    
    def __init__(self, alpha0:float=1.0,q:float=0.5):
        self.alpha, self.alpha0, self.q = alpha0, alpha0, q

    def __iter__(self):
        return self

    def __next__(self)->float:
        result = self.alpha
        self.alpha *= self.q
        return result

_PLOT_SPECS = (
    ("Error in x", ErrorMonitor, ("error_x",)),
    ("Error in y", ErrorMonitor, ("error_y",)),
    ("Duality gap", DualityGapStopping, ("duality gap",)),
    ("Optimality condition: dSstar", OptimalityCondStopping, ("dSstar",)),
    ("Optimality condition: dR", OptimalityCondStopping, ("dR",)),
)


def _rule_components(rule: StopRule) -> Iterator[StopRule]:
    """Yield a rule and all components of nested combined rules."""
    seen: set[int] = set()
    pending = [rule]
    while pending:
        component = pending.pop()
        identifier = id(component)
        if identifier in seen:
            continue
        seen.add(identifier)
        yield component
        children = getattr(component, "rules", ())
        if isinstance(children, Sequence):
            pending.extend(reversed(children))


def _history_steps(rule: StopRule, key: str, length: int) -> np.ndarray:
    """Return the iteration numbers belonging to one history."""
    step_histories = getattr(rule, "history_step_dict", {})
    steps = step_histories.get(key)
    if steps is None:
        return np.arange(length)
    if len(steps) != length:
        raise ValueError(
            f"History {key!r} of {rule!r} has {length} values but "
            f"{len(steps)} recorded iteration numbers."
        )
    return np.asarray(steps, dtype=int)


def _timing_components(components: Sequence[StopRule]) -> list[CountIterations]:
    """Return iteration counters that contain a nonempty timing history."""
    return [
        component
        for component in components
        if (
            isinstance(component, CountIterations)
            and getattr(component, "timing", False)
            and len(component.history_dict.get("time", ())) > 0
        )
    ]


def _times_for_history(
    rule: StopRule,
    key: str,
    length: int,
    clocks: Sequence[CountIterations],
) -> np.ndarray:
    """Align one rule history with an elapsed-time history."""
    target_steps = _history_steps(rule, key, length)

    for clock in clocks:
        times = np.asarray(clock.history_dict["time"], dtype=float)
        clock_steps = getattr(clock, "history_step_dict", {}).get("time")
        if clock_steps is not None:
            if len(clock_steps) != len(times):
                continue
            time_at_step = dict(zip(clock_steps, times, strict=True))
            if all(int(step) in time_at_step for step in target_steps):
                return np.asarray(
                    [time_at_step[int(step)] for step in target_steps],
                    dtype=float,
                )
        elif target_steps.size == 0:
            return np.asarray([], dtype=float)
        elif np.all(target_steps >= 0) and np.max(target_steps) < len(times):
            return times[target_steps]
        elif len(times) >= length:
            # Compatibility with older stop rules without history_step_dict.
            return times[:length]

    raise ValueError(
        f"Could not align history {key!r} of {rule!r} with the elapsed-time "
        "history of a CountIterations(timing=True) component."
    )


def convergence_plots(
    setting: Setting,
    plot_over_time: bool = False,
    methods: list[str] | None = None,
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot convergence histories for selected methods that have been run.

    The function inspects the stopping rule stored for every method in
    ``setting._methods``. Nested combined rules are searched recursively for
    :class:`ErrorMonitor`, :class:`DualityGapStopping`, and
    :class:`OptimalityCondStopping` components. Each available history type
    gets its own subplot, and every recorded history is plotted on a
    logarithmic y-axis.

    Parameters
    ----------
    setting : regpy.solvers.Setting
        Setting whose method histories should be plotted.
    plot_over_time : bool, optional
        Plot against iteration numbers if ``False`` (default). If ``True``,
        plot against CPU time recorded by a ``CountIterations(timing=True)``
        component in each method's stopping rule.
    methods : list of str or None, optional
        Short names of methods to plot. If omitted, all methods with recorded
        histories are plotted. The order of the list determines the order of
        curves in each legend.
    title : str or None, optional
        Figure title. If omitted, no figure-level title is added.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The created figure.
    axes : numpy.ndarray
        One-dimensional array containing the created subplot axes.

    Raises
    ------
    TypeError
        If ``plot_over_time`` is not boolean or the setting has no method
        dictionary, or if ``methods`` or ``title`` has the wrong type.
    ValueError
        If a requested method name is unknown, if no requested histories are
        available, or if time plotting was requested but a history cannot be
        aligned with a timing counter.
    """
    import matplotlib.pyplot as plt    
    if not isinstance(plot_over_time, (bool, np.bool_)):
        raise TypeError("plot_over_time must be boolean.")

    method_registry = getattr(setting, "_methods", None)
    if not isinstance(method_registry, dict):
        raise TypeError("setting must provide a dictionary attribute _methods.")
    if methods is not None:
        if not isinstance(methods, list) or not all(
            isinstance(method, str) for method in methods
        ):
            raise TypeError("methods must be a list of strings or None.")
        unknown_methods = [
            method for method in methods if method not in method_registry
        ]
        if unknown_methods:
            raise ValueError(
                "Unknown method name(s): " + ", ".join(unknown_methods)
            )
        selected_methods = [
            (method, method_registry[method]) for method in methods
        ]
    else:
        selected_methods = list(method_registry.items())
    if title is not None and not isinstance(title, str):
        raise TypeError("title must be a string or None.")

    series: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {
        subplot_title: [] for subplot_title, _, _ in _PLOT_SPECS
    }

    for method_name, method_data in selected_methods:
        if not isinstance(method_data, dict):
            continue
        stoprule = method_data.get("stoprule")
        if not isinstance(stoprule, StopRule):
            continue

        # Do not use ``stoprule.solver`` to decide whether the method ran.
        # TikhonovCG combines the setting's rule with its internal rules.
        # CombineRules flattens that nested rule, so the root retained by the
        # setting is not attached to the solver, although its child rules are
        # attached and contain the recorded histories.
        components = list(_rule_components(stoprule))
        has_requested_history = any(
            isinstance(component, rule_type)
            and any(
                len(component.history_dict.get(key, ())) > 0
                for key in history_keys
            )
            for _, rule_type, history_keys in _PLOT_SPECS
            for component in components
        )
        if not has_requested_history:
            continue

        clocks = _timing_components(components) if plot_over_time else []
        if plot_over_time and not clocks:
            raise ValueError(
                f"Method {method_name!r} has been run but its stopping rule "
                "has no CountIterations(timing=True) component with timing "
                "data."
            )

        for subplot_title, rule_type, history_keys in _PLOT_SPECS:
            matching_rules = [
                component
                for component in components
                if isinstance(component, rule_type)
            ]
            for rule_index, component in enumerate(matching_rules, start=1):
                rule_suffix = (
                    f" #{rule_index}" if len(matching_rules) > 1 else ""
                )
                for key in history_keys:
                    values = np.asarray(
                        component.history_dict.get(key, ()), dtype=float
                    )
                    if values.ndim != 1:
                        raise ValueError(
                            f"History {key!r} of method {method_name!r} must "
                            "be one-dimensional."
                        )
                    if values.size == 0:
                        continue
                    horizontal = (
                        _times_for_history(
                            component, key, len(values), clocks
                        )
                        if plot_over_time
                        else _history_steps(component, key, len(values))
                    )
                    label = f"{method_name}{rule_suffix}: {key}"
                    series[subplot_title].append(
                        (label, horizontal, values)
                    )

    active_series = [
        (subplot_title, curves)
        for subplot_title, curves in series.items()
        if curves
    ]
    if not active_series:
        raise ValueError(
            "No recorded ErrorMonitor, DualityGapStopping, or "
            "OptimalityCondStopping histories were found."
        )

    figure, axes_grid = plt.subplots(
        len(active_series),
        1,
        figsize=(8.0, 3.4 * len(active_series)),
        squeeze=False,
        sharex=False,
    )
    axes: np.ndarray = axes_grid[:, 0]
    x_label = "CPU time" if plot_over_time else "Iteration"

    for axis, (subplot_title, curves) in zip(
        axes, active_series, strict=True
    ):
        axis.set_yscale("log")
        for label, horizontal, values in curves:
            axis.plot(horizontal, values, label=label)
        axis.set_title(subplot_title)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Value")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()

    if title is not None:
        figure.suptitle(title)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        figure.tight_layout()
    return figure, axes

"""Convergence plots; Matplotlib is imported only when plotting."""
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
    from regpy.solvers.setting import Setting
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING
import numpy as np
from regpy.stoprules import (
    CountIterations, DualityGapStopping, ErrorMonitor,
    OptimalityCondStopping, StopRule,
)

__all__ = ['convergence_plots', 'op_eval_over_time_plot', 'plot_operator_evaluations', 'get_method_colors']

_PLOT_SPECS = (
    ("Error in x", ErrorMonitor, ("error_x",)),
    ("Error in y", ErrorMonitor, ("error_y",)),
    ("Duality gap", DualityGapStopping, ("duality gap",)),
    ("Duality gap comparison", DualityGapStopping, ()),
    ("Optimality condition: dSstar", OptimalityCondStopping, ("dSstar",)),
    ("Optimality condition: dR", OptimalityCondStopping, ("dR",)),
)


def get_method_colors(
    setting: Setting, methods: list[str] | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    r"""Return persistent method colors for convergence and application plots.
    
    ``methods`` contains short names from ``setting._methods``; None selects
    all registered methods. On first use, colors are allocated in the requested
    order from Matplotlib's current color cycle. Later calls reuse assigned
    colors regardless of selection, order, or changes to the color cycle.
    The palette repeats when exhausted. New methods receive the next color.
    
    Colors and the initial palette are stored privately on the setting. The
    returned dictionary is a copy, with concrete RGBA tuples as values.
    No convergence plot needs to have been created first. Matplotlib is
    imported only when this helper is called.
    
    Application code can use::
    
        colors = get_method_colors(setting, list(reconstructions))
        ax.plot(grid, reconstructions[method], color=colors[method])
    
    Raises TypeError for an invalid registry or methods argument and
    ValueError for unknown method names.
    
    Parameters
    ----------
    setting: Setting
        Problem setting containing the operator, penalty, and data fidelity.
    methods: list[str] | None [default: None]
        Names of the implemented functional operations.
    
    
    Returns
    -------
    dict[str, tuple[float, float, float, float]]
        Result described above.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    registry = getattr(setting, "_methods", None)
    if not isinstance(registry, dict):
        raise TypeError("setting must provide a dictionary attribute _methods.")
    if methods is not None and (
        not isinstance(methods, list)
        or not all(isinstance(name, str) for name in methods)
    ):
        raise TypeError("methods must be a list of strings or None.")
    names = list(registry) if methods is None else methods
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise ValueError("Unknown method name(s): " + ", ".join(unknown))
    if not names:
        return {}
    state = getattr(setting, "_plot_method_color_state", None)
    if state is None:
        palette = plt.rcParams['axes.prop_cycle'].by_key().get('color') or ['C0']
        state = (tuple(to_rgba(color) for color in palette), {})
        setting._plot_method_color_state = state
    palette, colors = state
    for name in names:
        if name not in colors:
            colors[name] = palette[len(colors) % len(palette)]
    return {name: colors[name] for name in names}


def _history_keys(component: int, subplot_title: str, default_keys: list[str]) -> list[str]:
    r"""Resolve the dynamically named additional gap histories.
    
    Parameters
    ----------
    component: int
        Component.
    subplot_title: str
        Subplot title.
    default_keys: list[str]
        Default keys.
    
    
    Returns
    -------
    list[str]
        Result described above.
    """
    if subplot_title == "Duality gap comparison":
        return ("duality gap",) + tuple(getattr(component, "additional_gap_keys", ()))
    return default_keys


def _rule_components(rule: StopRule) -> Iterator[StopRule]:
    r"""Yield a rule and all components of nested combined rules.
    
    Parameters
    ----------
    rule: StopRule
        Rule.
    
    
    Yields
    ------
    StopRule
        Result described above.
    """
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
    r"""Return the iteration numbers belonging to one history.
    
    Parameters
    ----------
    rule: StopRule
        Rule.
    key: str
        Key.
    length: int
        Length.
    
    
    Returns
    -------
    np.ndarray
        Computed array.
    """
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
    r"""Return iteration counters that contain a nonempty timing history.
    
    Parameters
    ----------
    components: Sequence[StopRule]
        Components.
    
    
    Returns
    -------
    list[CountIterations]
        Result described above.
    """
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
    r"""Align one rule history with an elapsed-time history.
    
    Parameters
    ----------
    rule: StopRule
        Rule.
    key: str
        Key.
    length: int
        Length.
    clocks: Sequence[CountIterations]
        Clocks.
    
    
    Returns
    -------
    np.ndarray
        Computed array.
    """
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


def _evals_for_history(rule: StopRule, key: Any, length: int, components: list[str]) -> np.ndarray:
    r"""Match cumulative weighted evaluation counts by recorded solver step.
    
    Parameters
    ----------
    rule: StopRule
        Rule.
    key: Any
        Key.
    length: int
        Length.
    components: list[str]
        Components.
    
    
    Returns
    -------
    np.ndarray
        Computed array.
    """
    target_steps = _history_steps(rule, key, length)
    for counter in components:
        if not isinstance(counter, CountIterations):
            continue
        totals = getattr(counter, "eval_counts", {}).get("total", ())
        steps = getattr(counter, "eval_count_steps", ())
        if not totals or len(totals) != len(steps):
            continue
        by_step = dict(zip(steps, totals, strict=True))
        if all(int(step) in by_step for step in target_steps):
            return np.asarray([by_step[int(step)] for step in target_steps])
    raise ValueError(
        f"Cannot align history {key!r} with operator evaluation counts. "
        "Rerun with CountIterations(eval_count=True)."
    )


def convergence_plots(
    setting: Setting,
    x_axis: str = "its",
    methods: list[str] | None = None,
    title: str | None = None,
    fontsize:int =12,
    ref_err_x: float | None = None,
    ref_err_y: float | None = None,
    gap_pow: float = 2,
)  -> tuple[Any, Any]:
    r"""Plot convergence histories for selected methods that have been run.
    
    The function inspects the stopping rule stored for every method in
    ``setting._methods``. Nested combined rules are searched recursively for
    :class:`ErrorMonitor`, :class:`DualityGapStopping`, and
    :class:`OptimalityCondStopping` components. Each available history type
    gets its own subplot, and every recorded history is plotted on a
    logarithmic y-axis. The duality-gap subplot appears first and also shows
    the x-error histories as dashed lines and y-errors as dotted lines, both
    without markers. Y-errors are divided by sqrt(setting.regpar) only for
    convex Tikhonov settings (is_tikhonov and is_convex). These also share one errors panel.
    Methods with more than one recorded gap curve also appear in a separate
    comparison subplot containing their original and additional gaps, with
    one labelled curve per functional and method and the gap_pow transform.
    Only the first overlay of each error type is labelled in the gap panel.
    Final curve points are excluded from y-limit selection where possible,
    except in the errors panel, which uses automatic scaling of all points.
    Method colors are shared across subplots and retained on the setting.
    Application plots can retrieve them with ``get_method_colors(setting)``. Curve widths and marker sizes
    are half the Matplotlib defaults active when this function is called.
    
    Parameters
    ----------
    setting: Setting
        Setting whose method histories should be plotted.
    x_axis: str [default: 'its']
        Horizontal axis: iteration numbers, elapsed CPU time, or
        cumulative weighted operator evaluations. Time requires a
        CountIterations(timing=True) history; op_evals requires
        CountIterations(eval_count=True). Histories are aligned by solver
        iteration number. Normal-operator calls count twice. Counts exclude
        solver construction before the counting rule was attached.
    methods: list[str] | None [default: None]
        Short names of methods to plot. If omitted, all methods with recorded
        histories are plotted. The order of the list determines the order of
        curves in each legend.
    title: str | None [default: None]
        Figure title. If omitted, no figure-level title is added.
    fontsize: int [default: 12]
        Fontsize.
    ref_err_x: float | None [default: None]
        Add reference lines to the merged errors subplot where the respective
        error history exists. The x reference is dashed; the y reference is
        dotted and scaled by the same factor as the y-error curves (1/sqrt(regpar)
        only for convex Tikhonov settings).
        Values should be positive to be visible on the logarithmic y-axis.
        None omits the respective reference line.
    ref_err_y: float | None [default: None]
        Add reference lines to the merged errors subplot where the respective
        error history exists. The x reference is dashed; the y reference is
        dotted and scaled by the same factor as the y-error curves (1/sqrt(regpar)
        only for convex Tikhonov settings).
        Values should be positive to be visible on the logarithmic y-axis.
        None omits the respective reference line.
    gap_pow: float [default: 2]
        Positive finite exponent denominator for duality-gap curves. Plot
        gap**(1/gap_pow) for both gap panels, with matching subplot titles. Stored histories
        and all other subplot values are unchanged.
    
    Returns
    -------
    figure : matplotlib.figure.Figure
        The created figure.
    axes : numpy.ndarray
        One-dimensional array containing the created subplot axes.
    
    Raises
    ------
    TypeError
        If ``x_axis`` is not a string or the setting has no method
        dictionary, or if ``methods`` or ``title`` has the wrong type.
    ValueError
        If a requested method name is unknown, if no requested histories are
        available, if x_axis is unknown, or if a history cannot be aligned
        with the requested timing or evaluation counter.
    """
    if (isinstance(gap_pow, (bool, np.bool_))
            or not isinstance(gap_pow, (int, float, np.integer, np.floating))):
        raise TypeError("gap_pow must be a positive finite real number.")
    if not np.isfinite(gap_pow) or gap_pow <= 0:
        raise ValueError("gap_pow must be positive and finite.")
    import matplotlib.pyplot as plt    
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    if not isinstance(x_axis, str):
        raise TypeError("x_axis must be a string.")
    if x_axis not in {"its", "time", "op_evals"}:
        raise ValueError("x_axis must be 'its', 'time', or 'op_evals'.")

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

    curve_methods: dict[str, str] = {}
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
                for key in _history_keys(component, subplot_title, history_keys)
            )
            for subplot_title, rule_type, history_keys in _PLOT_SPECS
            for component in components
        )
        if not has_requested_history:
            continue

        clocks = _timing_components(components) if x_axis == "time" else []
        if x_axis == "time" and not clocks:
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
                for key in _history_keys(component, subplot_title, history_keys):
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
                        if x_axis == "time"
                        else _evals_for_history(component, key, len(values), components)
                        if x_axis == "op_evals"
                        else _history_steps(component, key, len(values))
                    )
                    label = f"{method_name}{rule_suffix}: {key}"
                    curve_methods[label] = method_name
                    series[subplot_title].append(
                        (label, horizontal, values)
                    )

    # A comparison is useful only for methods with multiple recorded gaps.
    comparison = series["Duality gap comparison"]
    gap_counts = {}
    for label, _, _ in comparison:
        method = curve_methods[label]
        gap_counts[method] = gap_counts.get(method, 0) + 1
    series["Duality gap comparison"] = [
        curve for curve in comparison if gap_counts[curve_methods[curve[0]]] > 1
    ]

    error_curves = []
    scale_y = bool(setting.is_tikhonov and setting.is_convex)
    y_scale = 1.0
    y_label = r"error_y/$\sqrt{\alpha}$" if scale_y else "error_y"
    if series["Error in y"] and scale_y:
        alpha = getattr(setting, "regpar", None)
        if (alpha is None or not np.isscalar(alpha) or not np.isrealobj(alpha)
                or not np.isfinite(alpha) or alpha <= 0):
            raise ValueError("Plotting error_y/sqrt(alpha) requires a positive finite setting.regpar.")
        y_scale = np.sqrt(alpha)
    for component, style, suffix in (
        ("Error in x", "--", "error_x"),
        ("Error in y", ":", y_label),
    ):
        for label, horizontal, values in series[component]:
            scaled = values / y_scale if component == "Error in y" else values
            error_curves.append((label, horizontal, scaled, style, suffix))
    active_series = [
        (name, curves) for name, curves in series.items()
        if curves and name not in ("Error in x", "Error in y")
    ]
    if error_curves:
        active_series.append(("Errors in x and y", []))
    if not active_series:
        raise ValueError(
            "No recorded ErrorMonitor, DualityGapStopping, or "
            "OptimalityCondStopping histories were found."
        )

    # Keep each method's color stable across histories and subplots.
    active_series.sort(key=lambda item: item[0] != "Duality gap")
    method_colors = get_method_colors(setting, [name for name, _ in selected_methods])
    linewidth = plt.rcParams['lines.linewidth'] / 2
    markersize = plt.rcParams['lines.markersize'] / 2

    figure, axes_grid = plt.subplots(
        len(active_series),
        1,
        figsize=(8.0, 3.4 * len(active_series)),
        squeeze=False,
        sharex=False,
    )
    axes: np.ndarray = axes_grid[:, 0]
    x_label = {"its": "Iteration", "time": "CPU time [s]",
               "op_evals": "Operator evaluations (weighted)"}[x_axis]

    for axis, (subplot_title, curves) in zip(
        axes, active_series, strict=True
    ):
        axis.set_yscale("log")
        axis.tick_params(axis="both", labelsize=fontsize-2)
        scale_samples = []

        def include_for_scaling(values: np.ndarray) -> None:
            # Keep all plotted points, but ignore terminal drops/spikes when
            # choosing limits. A singleton has no earlier point to use.
            values = np.asarray(values, dtype=float)
            values = values[:-1] if values.size > 1 else values
            valid = values[np.isfinite(values) & (values > 0)]
            if valid.size:
                scale_samples.append(valid)

        for curve_index, (label, horizontal, values) in enumerate(curves):
            plotted_values = (np.power(values, 1.0 / gap_pow)
                              if subplot_title in ("Duality gap", "Duality gap comparison") else values)
            style = (('-', '--', ':', '-.')[curve_index % 4]
                     if subplot_title == "Duality gap comparison" else '-')
            marker = (('.', 's', 'o','x')[curve_index % 4]
                      if subplot_title == "Duality gap comparison" else 'o')
            axis.plot(horizontal, plotted_values, marker=marker, linestyle=style, label=label,
                      color=method_colors[curve_methods[label]],
                      linewidth=linewidth, markersize=markersize)
            include_for_scaling(plotted_values)
        if subplot_title in ("Duality gap", "Errors in x and y"):
            labelled = set()
            for label, horizontal, values, style, suffix in error_curves:
                if subplot_title == "Duality gap":
                    legend_label = suffix if suffix not in labelled else '_nolegend_'
                    labelled.add(suffix)
                else:
                    legend_label = label.rsplit(': ', 1)[0] + ': ' + suffix
                axis.plot(horizontal, values, linestyle=style, marker=None,
                          color=method_colors[curve_methods[label]],
                          linewidth=linewidth, label=legend_label)
                include_for_scaling(values)
        if subplot_title == "Errors in x and y":
            references = []
            if series["Error in x"] and ref_err_x is not None:
                references.append((ref_err_x, '--', 'reference error_x'))
            if series["Error in y"] and ref_err_y is not None:
                references.append((ref_err_y / y_scale, ':', 'reference ' + y_label))
            for value, style, label in references:
                axis.axhline(value, color='black', linestyle=style,
                             label=label, linewidth=linewidth)
                include_for_scaling([value])
        if scale_samples and subplot_title != "Errors in x and y":
            samples = np.concatenate(scale_samples)
            lower, upper = np.log10(samples.min()), np.log10(samples.max())
            margin = max(0.05 * (upper - lower), 0.05)
            axis.set_ylim(10.0 ** (lower - margin), 10.0 ** (upper + margin))
        display_title = (
            rf"$(\mathrm{{Duality\ gap}})^{{1/{gap_pow:g}}}$"
            if subplot_title == "Duality gap" else subplot_title
        )
        if subplot_title == "Duality gap comparison":
            display_title = rf"$(\mathrm{{Duality\ gap\ comparison}})^{{1/{gap_pow:g}}}$"
        axis.set_title(display_title, fontsize=fontsize+2)
        axis.set_xlabel(x_label,fontsize = fontsize+2)
        axis.set_ylabel("Value",fontsize = fontsize+2)
        axis.grid(True, which="both", alpha=0.3,
                  linewidth=plt.rcParams["grid.linewidth"] / 2)
        axis.legend(fontsize = fontsize-2)

    if title is not None:
        figure.suptitle(title,fontsize = fontsize+2)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        figure.tight_layout()
    return figure, axes


def op_eval_over_time_plot(
    setting: Setting,
    methods: list[str] | None = None,
    title: str | None = None,
) -> tuple[Any, Any]:
    r"""Plot cumulative weighted operator evaluations against elapsed CPU time.
    
    Parameters
    ----------
    setting: Setting
        Setting containing executed stopping-rule histories.
    methods: list[str] | None [default: None]
        Short method names in plotting order. None selects all executed
        methods. Selected methods that have not run are skipped.
    title: str | None [default: None]
        Optional figure title.
    
    Returns
    -------
    figure : matplotlib.figure.Figure
        Created figure.
    axis : matplotlib.axes.Axes
        Axes containing one curve per method, with linear scales.
    
    Notes
    -----
    Requires CountIterations histories with timing=True and eval_count=True.
    Counts and times are matched by solver iteration, using only steps at
    which both are recorded (no interpolation). Normal-operator applications
    count twice. Counts exclude work before the counter was attached.
    Missing or unalignable histories raise ValueError; rerun the solver with
    both recording options enabled. Methods and title follow the validation
    rules of convergence_plots.
    """
    import matplotlib.pyplot as plt
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

    curves = []
    for method_name, method_data in selected_methods:
        if not isinstance(method_data, dict):
            continue
        rule = method_data.get("stoprule")
        if not isinstance(rule, StopRule):
            continue
        components = list(_rule_components(rule))
        clocks = _timing_components(components)
        aligned = None
        for counter in components:
            if not isinstance(counter, CountIterations):
                continue
            totals = getattr(counter, "eval_counts", {}).get("total", ())
            steps = getattr(counter, "eval_count_steps", ())
            if len(totals) == 0 or len(totals) != len(steps):
                continue
            for clock in clocks:
                times = np.asarray(clock.history_dict["time"], dtype=float)
                time_steps = _history_steps(clock, "time", len(times))
                time_at_step = dict(zip(time_steps, times, strict=True))
                pairs = [(time_at_step[step], total)
                         for step, total in zip(steps, totals, strict=True)
                         if step in time_at_step]
                if pairs:
                    aligned = np.asarray(pairs)
                    break
            if aligned is not None:
                break
        if aligned is None:
            raise ValueError(
                f"Method {method_name!r} has no aligned CPU-time and evaluation "
                "histories. Rerun with CountIterations(timing=True, eval_count=True)."
            )
        curves.append((method_name, aligned))
    if not curves:
        raise ValueError("No recorded operator-evaluation histories were found.")
    figure, axis = plt.subplots(figsize=(8., 4.5))
    method_colors = get_method_colors(setting, [name for name, _ in curves])
    for method_name, values in curves:
        axis.plot(values[:, 0], values[:, 1], 'o-', label=method_name,
                  color=method_colors[method_name])
    axis.set_xlabel("CPU time [s]")
    axis.set_ylabel("Operator evaluations (weighted)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()
    return figure, axis


# Preserve the previously advertised public name.
plot_operator_evaluations = op_eval_over_time_plot

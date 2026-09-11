"""Tests for the general stopping-rule infrastructure."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from regpy.stoprules.general import (
    AndCombineRules,
    CombineRules,
    CountIterations,
    MissingValueError,
    NoneRule,
    StopRule,
)


_MISSING = object()


class DummySolver:
    """Minimal solver interface needed by the general stopping rules."""

    def __init__(
        self,
        x=_MISSING,
        y=_MISSING,
        *,
        op=None,
        converged=False,
    ):
        if x is not _MISSING:
            self.x = x
        if y is not _MISSING:
            self.y = y
        if op is not None:
            self.op = op
        self._converged = converged

    def is_converged(self):
        return self._converged


class DecisionRule(StopRule):
    """Rule returning a prescribed sequence of decisions."""

    def __init__(self, decisions, *, history_name="decision"):
        super().__init__(logging_level=logging.CRITICAL)
        self.decisions = tuple(decisions)
        self.calls = 0
        self.history_dict = {history_name: []}
        self.history_name = history_name

    def _stop(self):
        index = min(self.calls, len(self.decisions) - 1)
        decision = self.decisions[index]
        self.calls += 1
        self.history_dict[self.history_name].append(float(decision))
        self.log_info = f"{self.history_name}:{decision}"
        return decision

    def reset(self):
        super().reset()
        self.calls = 0


class NeedsOperatorValue(StopRule):
    """Rule raising ``MissingValueError`` until ``solver.y`` exists."""

    def __init__(self, decision=False):
        super().__init__(logging_level=logging.CRITICAL)
        self.decision = decision
        self.seen_y = None

    def _stop(self):
        if self.solver is None or not hasattr(self.solver, "y"):
            raise MissingValueError("operator value unavailable")
        if self.solver.y is None:
            raise MissingValueError("operator value unavailable")
        self.seen_y = self.solver.y
        self.log_info = "operator value available"
        return self.decision


def _attach(rule, *, x=None, y=None, op=None, converged=False):
    solver = DummySolver(x=x, y=y, op=op, converged=converged)
    rule._complete_init_with_solver(solver)
    return solver


def test_stoprule_requires_stop_implementation():
    with pytest.raises(NotImplementedError):
        StopRule().stop()


def test_complete_init_attaches_solver():
    rule = DecisionRule([False])
    solver = DummySolver(x=np.array([1.0]))

    rule._complete_init_with_solver(solver)

    assert rule.solver is solver


def test_trigger_requires_attached_current_iterate():
    rule = DecisionRule([True])
    with pytest.raises(MissingValueError):
        rule.trigger()

    rule._complete_init_with_solver(DummySolver(x=None))
    with pytest.raises(MissingValueError):
        rule.trigger()


def test_main_rule_stores_current_iterate_when_triggered():
    x = np.array([1.0, 2.0])
    y = np.array([3.0])
    rule = DecisionRule([True])
    solver = _attach(rule, x=x, y=y)

    assert rule.stop()
    assert rule.triggered
    assert rule.x is solver.x
    assert rule.y is solver.y
    best_x, best_y = rule.best_iterate()
    assert best_x is solver.x
    assert best_y is solver.y


def test_subrule_does_not_store_iterate_automatically():
    rule = DecisionRule([True])
    rule.is_main_rule = False
    _attach(rule, x=np.array([1.0]), y=np.array([2.0]))

    assert rule.stop()
    assert not rule.triggered
    assert not hasattr(rule, "x")


def test_best_iterate_is_unavailable_before_trigger_or_convergence():
    rule = DecisionRule([False])
    _attach(rule, x=np.array([1.0]), y=np.array([2.0]))

    assert rule.best_iterate() is None


def test_best_iterate_uses_converged_solver_state():
    x = np.array([1.0])
    y = np.array([2.0])
    rule = DecisionRule([False])
    solver = _attach(rule, x=x, y=y, converged=True)

    best_x, best_y = rule.best_iterate()
    assert best_x is solver.x
    assert best_y is solver.y


def test_reset_clears_solver_trigger_and_histories():
    rule = DecisionRule([True], history_name="test_history")
    _attach(rule, x=np.array([1.0]), y=np.array([2.0]))
    assert rule.stop()

    rule.reset()

    assert rule.solver is None
    assert not rule.triggered
    assert not hasattr(rule, "x")
    assert not hasattr(rule, "y")
    assert rule.calls == 0
    assert rule.history_dict == {"test_history": []}


def test_copy_and_copy_and_reset_are_independent():
    rule = CountIterations(3, timing=True)
    _attach(rule, x=np.array([1.0]), y=np.array([2.0]))
    assert not rule.stop()

    copied = rule.copy()
    reset_copy = rule.copy_and_reset()

    assert copied is not rule
    assert copied.iteration == rule.iteration
    assert copied.history_dict is not rule.history_dict
    assert reset_copy.iteration == 0
    assert reset_copy.solver is None
    assert not reset_copy.triggered
    assert reset_copy.history_dict == {"time": []}


def test_none_rule_never_stops_and_returns_current_iterate():
    x = np.array([1.0])
    y = np.array([2.0])
    rule = NoneRule()
    solver = _attach(rule, x=x, y=y)

    assert not rule.stop()
    best_x, best_y = rule.best_iterate()
    assert best_x is solver.x
    assert best_y is solver.y


@pytest.mark.parametrize("invalid", (1.5, "3", None))
def test_count_iterations_requires_integer_limit(invalid):
    with pytest.raises(TypeError):
        CountIterations(invalid)


@pytest.mark.parametrize(
    "while_type, expected",
    (
        pytest.param(True, [False, False, True], id="while"),
        pytest.param(False, [False, True], id="until"),
    ),
)
def test_count_iterations_loop_semantics(while_type, expected):
    rule = CountIterations(
        2, while_type=while_type, logging_level=logging.CRITICAL
    )
    _attach(rule, x=np.array([1.0]), y=np.array([2.0]))

    decisions = [rule.stop() for _ in expected]

    assert decisions == expected
    assert rule.triggered
    assert rule.iteration == len(expected)


def test_count_iterations_zero_stops_immediately():
    rule = CountIterations(0, logging_level=logging.CRITICAL)
    _attach(rule, x=np.array([1.0]), y=np.array([2.0]))

    assert rule.stop()


def test_count_iterations_records_and_resets_timing(monkeypatch):
    # An explicit clock tests elapsed times without depending on how often
    # construction, attachment, and reset consult process_time().
    now = [10.0]
    monkeypatch.setattr(
        "regpy.stoprules.general.time.process_time",
        lambda: now[0],
    )
    rule = CountIterations(
        10, timing=True, logging_level=logging.CRITICAL
    )
    now[0] = 20.0
    _attach(rule, x=np.array([1.0]), y=np.array([2.0]))
    assert rule.start_time == 20.0

    now[0] = 20.25
    assert not rule.stop()
    now[0] = 20.75
    assert not rule.stop()
    assert rule.history_dict["time"] == pytest.approx([0.25, 0.75])

    now[0] = 22.0
    rule.reset()
    assert rule.history_dict == {"time": []}
    assert rule.start_time == 22.0

    now[0] = 30.0
    _attach(rule, x=np.array([1.0]), y=np.array([2.0]))
    now[0] = 30.5
    assert not rule.stop()
    assert rule.history_dict["time"] == pytest.approx([0.5])


@pytest.mark.parametrize("combiner", (CombineRules, AndCombineRules))
def test_combinations_validate_rule_sequence(combiner):
    with pytest.raises(TypeError):
        combiner(CountIterations(1))
    with pytest.raises(TypeError):
        combiner([CountIterations(1), object()])
    with pytest.raises(ValueError):
        combiner([])


def test_operator_overloads_construct_expected_combinations():
    left = DecisionRule([False])
    right = DecisionRule([False])
    assert isinstance(left + right, CombineRules)

    left = DecisionRule([False])
    right = DecisionRule([False])
    assert isinstance(left | right, CombineRules)

    left = DecisionRule([False])
    right = DecisionRule([False])
    assert isinstance(left & right, AndCombineRules)

    rule = DecisionRule([False])
    assert 0 + rule is rule
    with pytest.raises(TypeError):
        _ = 1 + rule


@pytest.mark.parametrize(
    "operator, combination_type",
    (
        pytest.param(lambda left, right: left | right, CombineRules, id="or"),
        pytest.param(
            lambda left, right: left & right,
            AndCombineRules,
            id="and",
        ),
    ),
)
def test_nested_combinations_of_same_kind_are_flattened(
    operator, combination_type
):
    rules = [DecisionRule([False]) for _ in range(3)]

    combination = operator(operator(rules[0], rules[1]), rules[2])

    assert isinstance(combination, combination_type)
    assert combination.rules == rules
    assert all(not rule.is_main_rule for rule in rules)


def test_sum_combines_rules_with_logical_or():
    rules = [DecisionRule([False]) for _ in range(3)]

    combination = sum(rules)

    assert isinstance(combination, CombineRules)
    assert combination.rules == rules


def test_or_combination_stops_when_first_rule_stops():
    first = DecisionRule([False, True], history_name="first")
    second = DecisionRule([False, False], history_name="second")
    combination = first | second
    solver = _attach(
        combination, x=np.array([1.0]), y=np.array([2.0])
    )

    assert not combination.stop()
    assert combination.stop()
    assert combination.triggered
    assert combination.active_rule is first
    best_x, best_y = combination.best_iterate()
    assert best_x is solver.x
    assert best_y is solver.y
    assert first.calls == second.calls == 2
    assert combination.history_dict["first"] == [0.0, 1.0]
    assert combination.history_dict["second"] == [0.0, 0.0]


def test_or_combination_uses_first_triggering_rule():
    first = DecisionRule([True], history_name="first")
    second = DecisionRule([True], history_name="second")
    combination = CombineRules([first, second])
    _attach(combination, x=np.array([1.0]), y=np.array([2.0]))

    assert combination.stop()
    assert combination.active_rule is first


def test_and_combination_stops_only_when_all_rules_stop():
    first = DecisionRule([True, True], history_name="first")
    second = DecisionRule([False, True], history_name="second")
    combination = first & second
    solver = _attach(
        combination, x=np.array([1.0]), y=np.array([2.0])
    )

    assert not combination.stop()
    assert combination.stop()
    assert combination.triggered
    best_x, best_y = combination.best_iterate()
    assert best_x is solver.x
    assert best_y is solver.y
    assert first.triggered
    assert first.calls == second.calls == 2


@pytest.mark.parametrize(
    "combine",
    (
        pytest.param(
            lambda rule: CombineRules([rule]), id="logical-or"
        ),
        pytest.param(
            lambda rule: AndCombineRules([rule]), id="logical-and"
        ),
    ),
)
def test_combination_computes_missing_operator_value(combine):
    rule = NeedsOperatorValue(decision=True)
    combination = combine(rule)
    solver = _attach(
        combination,
        x=np.array([1.0, 2.0]),
        y=None,
        op=lambda vector: 2 * vector,
    )

    assert combination.stop()
    np.testing.assert_array_equal(solver.y, np.array([2.0, 4.0]))
    np.testing.assert_array_equal(rule.seen_y, solver.y)


@pytest.mark.parametrize("combiner", (CombineRules, AndCombineRules))
def test_missing_operator_value_without_operator_is_runtime_error(combiner):
    combination = combiner([NeedsOperatorValue()])
    _attach(combination, x=np.array([1.0]), y=None)

    with pytest.raises(RuntimeError, match="operator value"):
        combination.stop()


@pytest.mark.parametrize("combiner", (CombineRules, AndCombineRules))
def test_combination_propagates_solver_to_all_subrules(combiner):
    rules = [DecisionRule([False]), DecisionRule([False])]
    combination = combiner(rules)
    solver = _attach(
        combination, x=np.array([1.0]), y=np.array([2.0])
    )

    assert combination.solver is solver
    assert all(rule.solver is solver for rule in rules)


@pytest.mark.parametrize("combiner", (CombineRules, AndCombineRules))
def test_combination_reset_resets_all_subrules(combiner):
    rules = [
        DecisionRule([True], history_name="first"),
        DecisionRule([True], history_name="second"),
    ]
    combination = combiner(rules)
    _attach(combination, x=np.array([1.0]), y=np.array([2.0]))
    assert combination.stop()

    combination.reset()

    assert combination.solver is None
    assert combination.active_rule is None
    assert not combination.triggered
    assert all(rule.solver is None for rule in rules)
    assert all(rule.calls == 0 for rule in rules)
    assert combination.history_dict == {"first": [], "second": []}


def test_combination_representations_show_component_rules():
    rule = CountIterations(2)

    assert repr(CombineRules([rule])).startswith("CombineRules([")
    assert repr(AndCombineRules([rule])).startswith("AndCombineRules([")

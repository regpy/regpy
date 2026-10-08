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
def test_nested_combinations_preserve_identity_and_root(
    operator, combination_type
):
    # A combination has its own schedule and result-selection state and must
    # therefore remain part of the tree, even inside the same combination type.
    rules = [DecisionRule([False, True], history_name=f'rule_{i}')
             for i in range(3)]
    inner = operator(rules[0], rules[1])
    combination = operator(inner, rules[2])

    assert isinstance(combination, combination_type)
    assert combination.rules == [inner, rules[2]]
    assert inner.rules == rules[:2]
    assert combination.is_main_rule
    assert all(not rule.is_main_rule for rule in [inner, *rules])
    assert all(rule.main_rule is combination for rule in [inner, *rules])
    solver = _attach(combination, x=np.array([1.0]), y=np.array([2.0]))
    assert all(rule.solver is solver for rule in [inner, *rules])
    assert not combination.stop()
    assert combination.stop()
    assert all(rule.calls == 2 for rule in rules)
    best_x, best_y = combination.best_iterate()
    assert best_x is solver.x
    assert best_y is solver.y


def test_sum_combines_rules_with_logical_or():
    rules = [DecisionRule([False]) for _ in range(3)]

    combination = sum(rules)

    assert isinstance(combination, CombineRules)
    assert combination.rules[0].rules == rules[:2]
    assert combination.rules[1] is rules[2]
    assert all(rule.main_rule is combination for rule in rules)
    _attach(combination, x=np.array([1.0]), y=np.array([2.0]))
    assert not combination.stop()
    assert all(rule.calls == 1 for rule in rules)
    rules[2].decisions = (True,)
    assert combination.stop()
    assert combination.active_rule is rules[2]


@pytest.mark.parametrize("combination_type", (CombineRules, AndCombineRules))
def test_nested_combination_keeps_its_own_schedule(combination_type):
    first = DecisionRule([False], history_name='first')
    second = DecisionRule([False], history_name='second')
    third = DecisionRule([False], history_name='third')
    inner = combination_type([first, second], schedule=[0, 2])
    outer = combination_type([inner, third])
    solver = _attach(outer, x=np.array([1.0]), y=np.array([2.0]))

    for step in range(3):
        solver.iteration_step_nr = step
        assert not outer.stop()
        assert first.calls == second.calls == (1 if step < 2 else 2)
        assert third.calls == step + 1
    assert inner.schedule is not None
    assert first.main_rule is outer
    assert second.main_rule is outer


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


# Evaluation histories are local to each rule; operator counters are shared.
def _counted_identity():
    from regpy.operators import Identity
    from regpy.vecsps import NumPyVectorSpace

    op = Identity(NumPyVectorSpace(3))
    op.enable_eval_count()
    return op, op.domain.ones()


def test_count_iterations_preparation_preserves_constructor_work():
    op, x = _counted_identity()
    op(x)  # Work before this run must not enter its history.
    rule = CountIterations(10, eval_count=True)
    rule.prepare_run(op)
    op(x)  # Simulate solver-constructor work.
    op.adjoint_eval(x)  # Weighted as two operator evaluations.
    _attach(rule, op=op, x=x)

    assert not rule.stop()
    assert rule.eval_counts['eval'] == [1]
    assert rule.eval_counts['adjoint_eval'] == [1]
    assert rule.eval_counts['total'] == [3]
    assert rule.eval_count_steps == [0]
    assert op.total_eval_count() == 4


def test_count_iterations_nested_runs_preserve_outer_history_and_cap():
    op, x = _counted_identity()
    op(x)
    outer = CountIterations(100, max_op_evals=5)
    _attach(outer, op=op, x=x)
    assert not outer.stop()
    op(x)

    # Two successive inner runs share the same operator and start at zero.
    for _ in range(2):
        inner = CountIterations(100, max_op_evals=2)
        inner.prepare_run(op)
        op(x)  # Count constructor work in both inner and outer histories.
        _attach(inner, op=op, x=x)
        assert not inner.stop()
        op(x)
        assert inner.stop()
        assert inner.eval_counts['total'] == [1, 2]
        stopped = outer.stop()

    assert stopped
    assert outer.eval_counts['total'] == [0, 3, 5]
    assert op.total_eval_count() == 6
    assert 'Operator evaluation cap' in outer.stop_reason


def test_count_iterations_reset_and_copy_do_not_reset_operator():
    op, x = _counted_identity()
    rule = CountIterations(10, eval_count=True)
    _attach(rule, op=op, x=x)
    op(x)
    assert not rule.stop()
    saved = rule.copy_and_reset()
    rule.reset()
    assert op.total_eval_count() == 1
    assert rule.eval_counts == {}
    assert saved.eval_counts == {}
    for current in (rule, saved):
        _attach(current, op=op, x=x)
        assert not current.stop()
        op(x)
        assert not current.stop()
        assert current.eval_counts['total'] == [0, 1]
    assert op.total_eval_count() == 3


def test_count_iterations_attachment_to_another_operator_rebases():
    first, x = _counted_identity()
    second, _ = _counted_identity()
    rule = CountIterations(10, eval_count=True)
    rule.prepare_run(first)
    first(x)
    second(x)
    _attach(rule, op=second, x=x)
    assert not rule.stop()
    second(x)
    assert not rule.stop()
    assert rule.eval_counts['total'] == [0, 1]
    assert first.total_eval_count() == 1
    assert second.total_eval_count() == 2


def test_count_iterations_scheduled_snapshots_use_weighted_baseline():
    op, x = _counted_identity()
    op.adjoint_eval(x)  # Historical work must not trigger the new cap.
    rule = CountIterations(10, max_op_evals=2, schedule=[0, 2])
    solver = _attach(rule, op=op, x=x)
    solver.iteration_step_nr = 0
    assert not rule.stop()
    op.adjoint_eval(x)
    solver.iteration_step_nr = 1
    assert not rule.stop()  # Omitted step still records its snapshot.
    solver.iteration_step_nr = 2
    assert rule.stop()
    assert rule.eval_counts['total'] == [0, 2, 2]
    assert rule.eval_counts['adjoint_eval'] == [0, 1, 1]
    assert rule.eval_count_steps == [0, 1, 2]


def test_count_iterations_disabled_leaves_existing_counting_unchanged():
    op, x = _counted_identity()
    op(x)
    rule = CountIterations(10)
    rule.prepare_run(op)
    _attach(rule, op=op, x=x)
    op(x)
    assert not rule.stop()
    assert rule.eval_counts == {}
    assert op.total_eval_count() == 2


def test_count_iterations_alpha_grid_keeps_cumulative_outer_counts():
    from regpy.hilbert import L2
    from regpy.operators import PtwMultiplication
    from regpy.solvers import Setting
    from regpy.solvers.linear.bregman import RunOnAlphaGrid
    from regpy.solvers.linear.proximal_gradient import FISTA
    from regpy.vecsps import NumPyVectorSpace

    space = NumPyVectorSpace(5)
    op = PtwMultiplication(space, np.linspace(.4, 1.2, 5))
    setting = Setting(op, L2, L2, data=space.ones(), logging_level='WARNING')
    method = {
        'class': FISTA, 'full': 'FISTA', 'primal': True,
        'method_params': {'op_norm': 1.2},
        'stoprule_params': {'max_it': 2, 'max_op_evals': None},
    }
    solver = RunOnAlphaGrid(setting, regpars=[.1, .05],
                            inner_solver=method, logging_level='WARNING')
    outer = CountIterations(10, eval_count=True)
    outer.prepare_run(op)
    outer._complete_init_with_solver(solver)
    # Ensure an inner reset would cause a visible decrease.
    for _ in range(100):
        op(solver.x)
    assert not outer.stop()
    for _ in range(2):
        solver.next()
        outer.stop()
        assert solver.inner_setting.op is op
        assert outer.eval_counts['total'][-1] == op.total_eval_count()
    assert outer.eval_counts['total'][0] == 100
    assert all(a < b for a, b in zip(outer.eval_counts['total'],
                                    outer.eval_counts['total'][1:]))

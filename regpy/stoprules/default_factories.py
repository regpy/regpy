"""Default stopping-rule selection without solver construction or evaluations.

All factories accept (setting, params). params may contain solver_class for
checking required solver accessors before constructing a rule. The selection
checks advertised capabilities, not whether a gap is finite at every iterate.
"""
from .general import CountIterations_factory
from .parameter_choice_rules import Discrepancy_factory
from .inner_stoprules import (
    RelativeChangeSol_factory, DualityGapStopping_factory,
    OptimalityCondStoppingNonConvex_factory,
)

__all__ = ['default_parameter_choice_rule', 'default_convex_stoprule',
           'default_nonconvex_stoprule', 'resolve_stoprule_factory']


def _has(functional, *methods):
    return all(name in getattr(functional, 'methods', ()) for name in methods)


def _solver_has(params, name):
    solver_class = params.get('solver_class')
    return solver_class is None or callable(getattr(solver_class, name, None))


def default_parameter_choice_rule(setting, params):
    """Use discrepancy with available positive absolute noise, else only a cap.

    A codomain-free setting always uses the cap. Discrepancy also requires
    observed data, either in the setting or in the supplied rule parameters.
    """
    options = params.get('rule_params', {})
    noise = options.get('noiselevel', params.get('absolute_noise_level',
            setting._noise_model_param.get('absolute_noise_level')))
    data = options.get('data', params.get('data', setting.data))
    if not setting.codomain_free and noise is not None and noise > 0 and data is not None:
        return Discrepancy_factory(setting, params)
    return CountIterations_factory(setting, params)


def default_convex_stoprule(setting, params):
    """Use an evaluable Fenchel gap, otherwise relative solution change.

    Check the functionals actually used by Setting.duality_gap. For a dual
    setting with a primal accessor, this is the original primal setting:
    the supplied tuple is an original dual pair and the missing primal pair
    is reconstructed using the original penalty's conjugate subgradient.
    No transformed double conjugates or subgradients are required on that path.

    For a standalone primal setting, two supplied pairs need no reconstruction;
    one supplied pair needs the appropriate subgradient. Accessor availability
    is checked on solver_class when supplied. This checks advertised methods,
    not finiteness of the gap or the values returned by solver accessors.
    """
    applicable = setting.is_tikhonov and setting.is_convex and not setting.codomain_free
    if applicable:
        primal = _solver_has(params, 'primal')
        dual = _solver_has(params, 'dual')
        original = getattr(setting, '_primal_setting', None)
        if original is not None and primal:
            # Matches the early delegation in Setting.duality_gap(primal=...).
            R, S = original.penalty, original.data_fid
            applicable = (original.is_tikhonov and original.is_convex
                          and not original.codomain_free
                          and all(_has(f, 'eval') for f in (R, S, R.conj, S.conj))
                          and _has(R.conj, 'subgradient'))
        else:
            R, S = setting.penalty, setting.data_fid
            applicable = all(_has(f, 'eval') for f in (R, S, R.conj, S.conj))
            if params.get('solver_class') is None:
                applicable = applicable and _has(S, 'subgradient') and _has(R.conj, 'subgradient')
            else:
                applicable = applicable and (
                    (primal and dual)
                    or (primal and _has(S, 'subgradient'))
                    or (dual and _has(R.conj, 'subgradient')))
    factory = DualityGapStopping_factory if applicable else RelativeChangeSol_factory
    return factory(setting, params)


def default_nonconvex_stoprule(setting, params):
    """Use optimality-condition violation when supported, else relative change.

    Require a nonconvex Tikhonov setting, explicit codomain, the fidelity's
    subgradient, the penalty's dist_subdiff, and the solver's primal accessor.
    The operator must support linearization as required by the Setting API.
    """
    applicable = (setting.is_tikhonov and not setting.is_convex
                  and not setting.codomain_free
                  and _has(setting.data_fid, 'subgradient')
                  and _has(setting.penalty, 'dist_subdiff')
                  and callable(getattr(setting.op, 'linearize', None))
                  and _solver_has(params, 'primal'))
    factory = OptimalityCondStoppingNonConvex_factory if applicable else RelativeChangeSol_factory
    return factory(setting, params)


def resolve_stoprule_factory(factory):
    """Resolve a callable, a public factory name, or its StopRule class name.

    Only factories explicitly exported by these modules are accepted; names
    are not evaluated as Python code. Examples: 'Discrepancy',
    'Discrepancy_factory', 'default_convex_stoprule'.
    """
    if callable(factory):
        return factory
    if not isinstance(factory, str):
        raise TypeError('factory must be callable or a registered name.')
    from . import general, inner_stoprules, parameter_choice_rules
    registry = {name: globals()[name] for name in __all__ if name.startswith('default_')}
    for module in (general, inner_stoprules, parameter_choice_rules):
        for name in module.__all__:
            if name.endswith('_factory'):
                registry[name] = getattr(module, name)
                registry[name[:-8]] = getattr(module, name)
    try:
        return registry[factory]
    except KeyError:
        raise ValueError(f'Unknown stoprule factory {factory!r}; choose from {sorted(registry)}.') from None

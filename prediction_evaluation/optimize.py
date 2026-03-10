from typing import Callable, Mapping, Sequence

from bayes_opt import BayesianOptimization

type _ParameterBound = tuple[float, float] | Sequence[str]


def optimize(
    function: Callable[..., float],
    param_bounds: Mapping[str, _ParameterBound],
    iterations: int,
) -> tuple[float, dict[str, float | str]]:
    optimizer = BayesianOptimization(
        f=function,
        # The docs confirm list-of-str is how you do categorical
        # https://bayesian-optimization.github.io/BayesianOptimization/3.2.0/parameter_types.html#3.-Categorical-variables
        pbounds=param_bounds,  # type: ignore[arg-type]
        random_state=1,
    )

    optimizer.maximize(n_iter=iterations)

    if optimizer.max is None:
        raise ValueError("Optimizer did not find a maximum")

    return optimizer.max["target"], optimizer.max["params"]

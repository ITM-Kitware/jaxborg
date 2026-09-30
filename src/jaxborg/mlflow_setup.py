"""Training adapter for JAXborg's MLflow provenance interface."""

from jaxborg.recipe import flatten_for_logging
from jaxborg.tracking import Run, configure  # noqa: F401


def start_run(recipe, *, backend, seed, effective_config=None, extra_tags=None, extra_params=None, name=None):
    run = Run(recipe, backend=backend, seed=seed, config=effective_config, tags=extra_tags, name=name)
    import mlflow

    params = {f"recipe.{k}": str(v)[:500] for k, v in flatten_for_logging(recipe).items() if v is not None}
    params.update({k: str(v)[:500] for k, v in (extra_params or {}).items()})
    mlflow.log_params(params)
    return run

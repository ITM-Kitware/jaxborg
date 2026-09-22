# Red/Blue training tests

This folder groups tests for Red/Blue learning and the cotraining workflow:

- JAX and CybORG trainers, PPO losses, joint environments, and learned-Red contracts.
- Feedforward MAPPO, recurrent IPPO, and LSTM MAPPO policies.
- Team recipe overrides, training topology sampling, and model bundles.
- Checkpoint evaluation, learned matchups, cross-play, cross-seed play, and post-training evaluation.

CLI argument handling and shell launchers are outside this folder's scope.
Dedicated CIA and resilience tests live in [`../resilience`](../resilience).

Shared fixtures and default slow-test exclusions come from `tests/conftest.py`
and `pyproject.toml`. General simulator, recipe-loader, and parity tests remain
in their existing locations.

Run the file relevant to your change from the repository root, for example:

```bash
uv run pytest tests/cotraining/test_recurrent_mappo.py
uv run pytest tests/cotraining/test_jax_joint_trainer.py
```

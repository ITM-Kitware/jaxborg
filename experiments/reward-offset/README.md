# Local joint payoff intervention

`rules.py` adds a fixed offset to each Blue reward and subtracts it from each
Red reward. It adds the same offset to `reward_ria`, keeping the trainer's raw
return accounting aligned with the payoff. This changes the scientific game.
It is a construction example, not a study result.

From Python, import `build_env` from this directory and bind explicit settings:

```python
from functools import partial

builder = partial(build_env, offset=7.0)
env, obs, state, init_states, update = make_joint_train(
    team_configs,
    networks,
    trainable_teams=("blue", "red"),
    env_factory=builder,
)
result = evaluate_matchup(
    blue_checkpoint,
    red_checkpoint,
    backend="jax",
    variant=variant,
    seeds=[42],
    env_factory=builder,
)
```

Both APIs construct outside compiled rollout. Omit `env_factory` for existing
construction. The callable receives `variant` and keyword arguments
`topology_mode`, `training_mode`, and `topology_path`. Training and evaluation
pass their own training mode and selected topology bank. Bind study settings in
the callable and pass the same callable to both APIs. Custom construction is not cached in evaluation contexts. Use immutable
settings so repeated calls retain the intended scientific behavior.

The supported contract is the joint CC4 JAX ABI: canonical ordered teams,
action meanings and dimensions, observation dimensions, simulator state and
constant pytrees, dtypes, and masks. Rules use explicit state and PRNG keys.
Reset and step must be pure and JAX traceable. Validation checks shapes and
structure before rollout; it cannot prove scientific equivalence, purity, or
reward semantics. Shared team rewards must be zero sum because the selected
matchup evaluator derives Red returns from Blue returns. Custom builders with
`red_reward` other than `zero_sum` are rejected. This example also rejects
resilience-role variants.

Override `step_env` for transitions and rewards so scalar auto-reset training,
batched training, and evaluation all invoke the rule. Overriding only `step`
would leave the other two paths unchanged. Reward components must continue to
sum to the Blue payoff for the trainer's raw-return metrics to be meaningful.
Custom observation encodings, action layouts, policy architectures and new
backends need separate support and are outside this hook.

`tests/cotraining/test_custom_environment.py` uses a deterministic one-tick
transition preserving the canonical state, observation and mask ABI with a
small policy. It runs actual compiled PPO and
learned matchup loops in scalar and batched modes with offsets three and seven,
checking payoff changes and structural rejection. The full simulator example
is separately traced against the canonical environment contract. Run from the
checkout with the shared environment:

```sh
JAX_PLATFORMS=cpu uv run pytest -n 0 tests/cotraining/test_custom_environment.py
```

For retained research execution, capture this module and all imported local
helper source, bound settings, dependency versions, inputs and variant alongside
the run. The callable name alone does not retain its implementation. This hook
does not implement execution-record capture.

# Learned Blue communication

## Implementation plan and scope

Implement the existing CC4 8-bit channel end to end: policy output, simulator
delivery, PPO training, checkpoints, evaluation, and communication ablations.
Only **Blue** sends or receives these messages. Red has no message head, action,
or observation features. Ordinary recipes and checkpoints default to messaging
off. Automatic enhanced-observation IOC sharing remains a separate mechanism.

The implementation follows these steps:

1. Add a strict top-level YAML boolean `use_messages`, default `false`. Resolve
   it into Blue's saved `arch.message_dim: 8`; leave Red unchanged and reject
   attempts to configure Red messages. Existing checkpoints without that field
   resolve to zero message dimensions.
2. Add eight independent Bernoulli outputs from the local actor features.
   JAX supports shared, separate, recurrent, MAPPO, and recurrent MAPPO policies.
   Native CybORG/Torch supports its existing shared and separate feedforward
   policies. Recurrent and MAPPO Torch architectures remain unsupported.
3. Deliver one vector per Blue sender to the other four Blue agents in the
   observation returned by the step. Messages are separate from game actions,
   so busy agents may send. Omitted messages clear the channel, and episode
   reset clears it. Both JAX wrappers and native CybORG use this contract.
4. Store sampled bits and the joint action/message log probability in rollout
   buffers. PPO uses the joint probability ratio. Busy Blue rows participate
   because they still choose messages, while their game-action mask permits
   only Sleep. `core.msg_ent_coef` controls message entropy regularization and
   defaults to `core.ent_coef`. Red's PPO behavior remains unchanged.
5. Save the message head and its dimension in bundles; use it during native
   CybORG and JAX evaluation. Add mute and foreign-sender controls, entropy,
   per-bit activation rates, and an observation-based message-use probe.
6. Verify real step-to-step native/JAX transport, reset behavior, checkpoint
   reloads, recurrent replay, and PPO updates to the message parameters.
   The previous message tests only injected received buffers; they did not
   exercise delivery.

## Configuration

```yaml
use_messages: false  # Default, including when omitted. Blue only.
```

Set it to `true` before training a new model. No observation dimensions change:
Blue already has 32 incoming message slots, eight from each teammate, excluding
itself. Enhanced models retain their existing 402/450 input layouts.

The actor decides the meaning of each bit through training. Unlike H-MARL's
handwritten host-alert encoder, this channel imposes no payload semantics.
Enabling it does not guarantee that agents learn useful communication.

JAX's distribution carries optional `message_logits`, preserving the existing
`(pi, value, carry)` policy helper interface. Message sampling uses an independent
key derived from the action key. With messages disabled, existing policy
parameters and action sampling are unchanged. Torch's existing action/value API
is also unchanged; `get_message_and_stats` supplies the optional message action.

Communication recipes are under `recipes/cotraining/messaging/`. For example:

```bash
uv run python scripts/train/algorithms/ippo_jax.py --recipe cotraining_mappo_comm --seed 42
uv run python scripts/train/algorithms/ippo_cyborg.py --recipe cotraining_comm --seed 42
```

`run_cotraining_mappo_comm.sh` launches the paired JAX MAPPO recipe seeds.
Long training and research sweeps are separate experiments, not prerequisites
for installing the feature; they are not launched by editing a recipe.

## Evaluation and protocol transfer

`eval_matchup.py` supports `--mute` and `--message-sender-path`, for both JAX and
Torch checkpoints. The actor checkpoint still supplies actions; a foreign Blue
checkpoint supplies outgoing bits from the same local observations, with its own
recurrent carry when applicable. Checkpoints must have matching observation
contracts. Muting affects only learned messages, not enhanced IOC alerts.

```bash
uv run python scripts/eval/eval_matchup.py --recipe cotraining_comm \
  --policy-backend jax --blue-path blue.safetensors --red-path red.safetensors --mute

uv run python scripts/eval/eval_comm_transfer.py --recipe cotraining_mappo_comm \
  --models seed42.safetensors seed43.safetensors --red-path fixed_red.safetensors \
  --seeds 1000,1001,1002 --output comm_transfer.jsonl
```

The transfer script evaluates every ordered pair of Blue checkpoints against
one fixed Red, using the same evaluation seeds and recipe topology bank. It
reports native, muted, and foreign-sender returns, `comm_gain = native - muted`,
`transfer = foreign - muted`, and `transfer_fraction = transfer / comm_gain`
(null when the denominator is near zero). Repeat for each training condition.

Training metrics include `team.blue.msg_entropy` and
`team.blue.msg_bit_mean_0` through `_7`. Matchup evaluation records per-episode
entropy, bit means, and per-bit mutual information with **observed** local
compromise evidence (IOCs for v2, retained evidence for v1, current alerts for
stock observations). The probe never reads latent Red compromise state. It is
a descriptive association, not proof that receivers use the messages or that
the protocol transfers. Native-versus-muted performance tests their usefulness;
foreign-sender comparisons test compatibility between independent runs.

## Verification

Run only the focused tests for this change:

```bash
uv run pytest -n 0 -s tests/test_blue_communication.py tests/cotraining/test_blue_message_training.py tests/cotraining/test_blue_message_evaluation.py
```

These include actual submissions through both environment step APIs, instead
of manually writing JAX's received-message buffer. Small PPO tests verify that
message parameters change even with all game actions forced to Sleep and no
message entropy bonus, and that frozen Red parameters stay unchanged.

## Research acceptance after training

Compare training curves and held-out native/muted/foreign returns across seeds.
A functioning transport and nonzero entropy do not establish useful cooperation.
Report whether enhanced IOC sharing was enabled in all compared arms. Existing
models trained without a message head require retraining to learn this channel.

"""Ray RLlib PPO baseline for CC4 — proper multi-agent with action masking."""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import mlflow
import numpy as np
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Agents.Wrappers import EnterpriseMAE
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from gymnasium.spaces import Box, Dict
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

warnings.filterwarnings("ignore", category=DeprecationWarning)

EXP_DIR = Path(os.environ.get("JAXBORG_EXP_DIR", "jaxborg-exp")).resolve()
NUM_AGENTS = 5
AGENT_IDS = [f"blue_agent_{i}" for i in range(NUM_AGENTS)]
POLICY_MAP = {agent_id: f"Agent{i}" for i, agent_id in enumerate(AGENT_IDS)}


class ActionMaskMAE(MultiAgentEnv):
    """Wraps EnterpriseMAE to put action_mask into the observation dict for Ray."""

    def __init__(self, env_config: dict | None = None):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=500,
        )
        cyborg = CybORG(scenario_generator=sg)
        self._env = EnterpriseMAE(cyborg)

        self._obs_spaces = {}
        self._act_spaces = {}
        for agent in AGENT_IDS:
            obs_space = self._env.observation_space(agent)
            act_space = self._env.action_space(agent)
            self._obs_spaces[agent] = Dict(
                {
                    "obs": obs_space,
                    "action_mask": Box(0.0, 1.0, shape=(act_space.n,), dtype=np.float32),
                }
            )
            self._act_spaces[agent] = act_space

        super().__init__()

    @property
    def agents(self):
        return self._env.agents

    def observation_space(self, agent):
        return self._obs_spaces[agent]

    def action_space(self, agent):
        return self._act_spaces[agent]

    def _wrap_obs(self, obs, info):
        wrapped = {}
        for agent in obs:
            mask = np.array(info[agent]["action_mask"], dtype=np.float32)
            wrapped[agent] = {"obs": obs[agent], "action_mask": mask}
        return wrapped

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset()
        return self._wrap_obs(obs, info), info

    def step(self, action_dict):
        obs, rew, term, trunc, info = self._env.step(action_dict=action_dict)
        return self._wrap_obs(obs, info), rew, term, trunc, info


def policy_mapper(agent_id, episode, worker, **kwargs):
    return POLICY_MAP[agent_id]


def train(
    total_timesteps,
    learning_rate,
    train_batch_size,
    minibatch_size,
    num_epochs,
    gamma,
    gae_lambda,
    clip_param,
    entropy_coeff,
    seed,
):
    register_env("CC4_masked", lambda config: ActionMaskMAE(config))

    # Build a temp env to get spaces
    tmp_env = ActionMaskMAE()

    config = (
        PPOConfig()
        .framework("torch")
        .environment(
            env="CC4_masked",
            action_mask_key="action_mask",
        )
        .debugging(seed=seed)
        .training(
            lr=learning_rate,
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            gamma=gamma,
            lambda_=gae_lambda,
            clip_param=clip_param,
            entropy_coeff=entropy_coeff,
            model={"fcnet_hiddens": [256, 256], "fcnet_activation": "tanh"},
        )
        .multi_agent(
            policies={
                ray_name: PolicySpec(
                    observation_space=tmp_env.observation_space(agent_id),
                    action_space=tmp_env.action_space(agent_id),
                )
                for agent_id, ray_name in POLICY_MAP.items()
            },
            policy_mapping_fn=policy_mapper,
        )
        .env_runners(
            num_env_runners=0,
            rollout_fragment_length=500,
        )
    )

    algo = config.build()

    # MLflow setup
    mlflow_db = EXP_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("ray-ppo-baseline")
    mlflow.start_run(run_name="ray-ppo-masked")
    mlflow.log_params(
        {
            "algorithm": "RayPPO-MultiAgent-Masked",
            "seed": seed,
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "train_batch_size": train_batch_size,
            "minibatch_size": minibatch_size,
            "num_epochs": num_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_param": clip_param,
            "entropy_coeff": entropy_coeff,
        }
    )

    save_dir = EXP_DIR / "ray_ppo"
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl"
    metrics_file = open(metrics_path, "w")

    start_time = time.perf_counter()
    total_steps = 0
    iteration = 0

    print(f"Training Ray PPO (multi-agent, action-masked) for {total_timesteps:,} timesteps", flush=True)
    print(f"  lr={learning_rate}, batch={train_batch_size}, entropy={entropy_coeff}", flush=True)

    while total_steps < total_timesteps:
        result = algo.train()
        iteration += 1
        total_steps = result.get("num_env_steps_sampled_lifetime", result.get("timesteps_total", 0))
        elapsed = time.perf_counter() - start_time

        # Extract metrics from Ray 2.38 result structure
        env_runners = result.get("env_runners", {})
        ep_reward_mean = env_runners.get("episode_reward_mean", 0.0)
        ep_len_mean = env_runners.get("episode_len_mean", 0.0)

        # Extract learner metrics from result["info"]["learner"][policy]["learner_stats"]
        info_learner = result.get("info", {}).get("learner", {})
        total_loss = 0.0
        policy_loss = 0.0
        vf_loss = 0.0
        entropy = 0.0
        vf_explained_var = 0.0
        n_found = 0
        for policy_name, policy_data in info_learner.items():
            ls = policy_data.get("learner_stats", {}) if isinstance(policy_data, dict) else {}
            if "total_loss" in ls:
                total_loss += ls["total_loss"]
                policy_loss += ls.get("policy_loss", 0.0)
                vf_loss += ls.get("vf_loss", 0.0)
                entropy += ls.get("entropy", 0.0)
                vf_explained_var += ls.get("vf_explained_var", 0.0)
                n_found += 1
        if n_found > 0:
            total_loss /= n_found
            policy_loss /= n_found
            vf_loss /= n_found
            entropy /= n_found
            vf_explained_var /= n_found

        sps = total_steps / elapsed if elapsed > 0 else 0

        record = {
            "iteration": iteration,
            "steps": total_steps,
            "wall_time_sec": elapsed,
            "steps_per_second": sps,
            "episode_reward_mean": ep_reward_mean,
            "episode_len_mean": ep_len_mean,
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "vf_loss": vf_loss,
            "entropy": entropy,
            "vf_explained_var": vf_explained_var,
        }
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

        mlflow_metrics = {
            k: float(v) for k, v in record.items() if isinstance(v, (int, float)) and k not in ("iteration",)
        }
        mlflow.log_metrics(mlflow_metrics, step=total_steps)

        print(
            f"  iter {iteration:3d} | steps {total_steps:>9,} | reward {ep_reward_mean:>8.1f} | "
            f"entropy {entropy:.3f} | expl_var {vf_explained_var:.4f} | {sps:.0f} sps",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    sps = total_steps / elapsed

    checkpoint_path = algo.save(str(save_dir / "checkpoint"))
    print(f"Saved checkpoint: {checkpoint_path}")

    mlflow.log_metrics(
        {
            "final_wall_time_sec": elapsed,
            "final_steps_per_second": sps,
            "final_episode_reward_mean": ep_reward_mean,
        }
    )
    mlflow.log_artifact(str(metrics_path))
    mlflow.end_run()
    metrics_file.close()
    algo.stop()

    print(f"\nDone! Wall time: {elapsed:.1f}s | Throughput: {sps:.0f} sps | Final reward: {ep_reward_mean:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Ray RLlib PPO multi-agent baseline (action-masked)")
    parser.add_argument("--total-timesteps", type=int, default=400_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-batch-size", type=int, default=2500)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(
        total_timesteps=args.total_timesteps,
        learning_rate=args.lr,
        train_batch_size=args.train_batch_size,
        minibatch_size=args.minibatch_size,
        num_epochs=args.num_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_param=args.clip_param,
        entropy_coeff=args.entropy_coeff,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

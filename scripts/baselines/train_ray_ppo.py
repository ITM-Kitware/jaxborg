"""Ray RLlib PPO baseline for CC4 — proper multi-agent with per-agent policies."""

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
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

warnings.filterwarnings("ignore", category=DeprecationWarning)

EXP_DIR = Path(__file__).resolve().parents[2].parent / "jaxborg-exp"
NUM_AGENTS = 5
POLICY_MAP = {f"blue_agent_{i}": f"Agent{i}" for i in range(NUM_AGENTS)}


def env_creator(env_config: dict):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(scenario_generator=sg)
    return EnterpriseMAE(cyborg)


def policy_mapper(agent_id, episode, worker, **kwargs):
    return POLICY_MAP[agent_id]


def train(total_timesteps, learning_rate, train_batch_size, sgd_minibatch_size,
          num_sgd_iter, gamma, gae_lambda, clip_param, entropy_coeff, seed):
    register_env("CC4", lambda config: env_creator(config))
    env = env_creator({})

    config = (
        PPOConfig()
        .framework("torch")
        .environment(env="CC4")
        .debugging(seed=seed)
        .training(
            lr=learning_rate,
            train_batch_size=train_batch_size,
            sgd_minibatch_size=sgd_minibatch_size,
            num_sgd_iter=num_sgd_iter,
            gamma=gamma,
            lambda_=gae_lambda,
            clip_param=clip_param,
            entropy_coeff=entropy_coeff,
            model={"fcnet_hiddens": [256, 256], "fcnet_activation": "tanh"},
        )
        .multi_agent(
            policies={
                ray_agent: PolicySpec(
                    policy_class=None,
                    observation_space=env.observation_space(cyborg_agent),
                    action_space=env.action_space(cyborg_agent),
                )
                for cyborg_agent, ray_agent in POLICY_MAP.items()
            },
            policy_mapping_fn=policy_mapper,
        )
        .env_runners(
            num_env_runners=0,  # single process for fair comparison
            rollout_fragment_length=500,
        )
    )

    algo = config.build()

    # MLflow setup
    mlflow_db = EXP_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("ray-ppo-baseline")
    mlflow.start_run(run_name="ray-ppo-multi-agent")
    mlflow.log_params({
        "algorithm": "RayPPO-MultiAgent",
        "seed": seed,
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "num_sgd_iter": num_sgd_iter,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_param": clip_param,
        "entropy_coeff": entropy_coeff,
    })

    save_dir = EXP_DIR / "ray_ppo"
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl"
    metrics_file = open(metrics_path, "w")

    start_time = time.perf_counter()
    total_steps = 0
    iteration = 0

    print(f"Training Ray PPO (multi-agent, 5 policies) for {total_timesteps:,} timesteps")
    print(f"  lr={learning_rate}, batch={train_batch_size}, entropy={entropy_coeff}")

    while total_steps < total_timesteps:
        result = algo.train()
        iteration += 1
        total_steps = result.get("timesteps_total", result.get("num_env_steps_sampled_lifetime", 0))
        elapsed = time.perf_counter() - start_time

        # Extract reward info
        env_runners = result.get("env_runners", {})
        ep_reward_mean = env_runners.get("episode_reward_mean", 0.0)
        ep_len_mean = env_runners.get("episode_len_mean", 0.0)

        learner = result.get("learner", {})
        # Aggregate across policies
        total_loss = 0.0
        policy_loss = 0.0
        vf_loss = 0.0
        entropy = 0.0
        n_policies = 0
        for policy_name, policy_stats in learner.items():
            if isinstance(policy_stats, dict) and "total_loss" in policy_stats:
                total_loss += policy_stats["total_loss"]
                policy_loss += policy_stats.get("policy_loss", 0.0)
                vf_loss += policy_stats.get("vf_loss", 0.0)
                entropy += policy_stats.get("entropy", 0.0)
                n_policies += 1
        if n_policies > 0:
            total_loss /= n_policies
            policy_loss /= n_policies
            vf_loss /= n_policies
            entropy /= n_policies

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
        }
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

        mlflow_metrics = {k: float(v) for k, v in record.items()
                         if isinstance(v, (int, float)) and k not in ("iteration",)}
        mlflow.log_metrics(mlflow_metrics, step=total_steps)

        if iteration % 10 == 0 or total_steps >= total_timesteps:
            print(f"  iter {iteration} | steps {total_steps:,} | reward {ep_reward_mean:.1f} | "
                  f"entropy {entropy:.2f} | {sps:.0f} sps")

    elapsed = time.perf_counter() - start_time
    sps = total_steps / elapsed

    # Save checkpoint
    checkpoint_path = algo.save(str(save_dir / "checkpoint"))
    print(f"Saved checkpoint: {checkpoint_path}")

    mlflow.log_metrics({
        "final_wall_time_sec": elapsed,
        "final_steps_per_second": sps,
        "final_episode_reward_mean": ep_reward_mean,
    })
    mlflow.log_artifact(str(metrics_path))
    mlflow.end_run()
    metrics_file.close()

    algo.stop()

    print(f"\nTraining complete!")
    print(f"Wall time: {elapsed:.1f}s")
    print(f"Throughput: {sps:.0f} sps")
    print(f"Final reward: {ep_reward_mean:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Ray RLlib PPO multi-agent baseline for CC4")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-batch-size", type=int, default=2500)
    parser.add_argument("--sgd-minibatch-size", type=int, default=256)
    parser.add_argument("--num-sgd-iter", type=int, default=4)
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
        sgd_minibatch_size=args.sgd_minibatch_size,
        num_sgd_iter=args.num_sgd_iter,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_param=args.clip_param,
        entropy_coeff=args.entropy_coeff,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

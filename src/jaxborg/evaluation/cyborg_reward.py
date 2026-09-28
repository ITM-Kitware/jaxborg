"""Read-only CIA reward accumulation for native CybORG evaluation."""

import random

from jaxborg.scenarios.cc4.topology_roles import ROLE_AUTH, ROLE_DB, ROLE_WEB, assign_resilience_roles


class CyborgRewardTracker:
    def __init__(self, env, seed, role_map=None):
        self.env = env
        if role_map is None:
            try:
                role_map = assign_resilience_roles(
                    env.unwrapped.environment_controller.state.hosts, random.Random(seed)
                )
            except ValueError:
                role_map = {}
        self.role_map = role_map
        hosts = env.unwrapped.environment_controller.state.hosts
        valid = sorted(role_map.values()) == [ROLE_AUTH, ROLE_DB, ROLE_WEB] and all(name in hosts for name in role_map)
        self.cia_sum = [0.0, 0.0, 0.0] if valid else None

    def step(self):
        if self.cia_sum is None:
            return
        hosts = self.env.unwrapped.environment_controller.state.hosts
        for name, role in self.role_map.items():
            impacted = any(
                service.get_service_reliability() < 100
                or (str(key).split(".")[-1] == "OTSERVICE" and not service.active)
                for key, service in hosts[name].services.items()
            )
            if impacted:
                for index, roles in enumerate(
                    ((ROLE_AUTH, ROLE_DB), (ROLE_AUTH, ROLE_WEB), (ROLE_AUTH, ROLE_DB, ROLE_WEB))
                ):
                    if role in roles:
                        self.cia_sum[index] -= 10.0


def native_reward_fields(default_returns, cia_sums, recipe):
    from jaxborg.evaluation.reward_reporting import reward_return_fields
    from jaxborg.reward_config import RewardConfig

    config = RewardConfig.from_recipe(recipe)
    shaped = [
        None if cia is None else float(score) + config.scale * sum(w * v for w, v in zip(config.weights, cia))
        for score, cia in zip(default_returns, cia_sums, strict=True)
    ]
    return reward_return_fields(default_returns, shaped, config=config)

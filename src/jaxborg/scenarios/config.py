from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioConfig:
    num_hosts: int
    num_subnets: int
    num_blue_agents: int
    num_red_agents: int
    num_services: int
    num_decoy_types: int
    mission_phases: int
    max_steps: int
    message_length: int
    max_detection_randoms: int
    num_green_random_fields: int
    num_red_policy_random_fields: int
    min_hosts_per_subnet: int
    max_hosts_per_subnet: int
    max_user_hosts: int
    max_server_hosts: int
    blue_max_observed_subnets: int
    max_tracked_session_pids: int
    max_tracked_suspicious_pids: int

    @property
    def total_action_actor_slots(self) -> int:
        return self.num_blue_agents + self.num_hosts + self.num_red_agents

    @property
    def obs_hosts_per_subnet(self) -> int:
        # Includes router for action encoding.
        return self.max_user_hosts + self.max_server_hosts + 1

    @property
    def obs_vector_hosts_per_subnet(self) -> int:
        # Excludes router — matches CybORG BlueFlatWrapper observation layout.
        return self.max_user_hosts + self.max_server_hosts

    @property
    def action_host_slots(self) -> int:
        return self.num_subnets * self.obs_hosts_per_subnet

    @property
    def blue_action_host_slots(self) -> int:
        return self.blue_max_observed_subnets * self.obs_vector_hosts_per_subnet

    @property
    def blue_traffic_slots(self) -> int:
        return (self.num_subnets - 1) * self.blue_max_observed_subnets

    @property
    def num_messages(self) -> int:
        return self.num_blue_agents - 1

    @property
    def blue_obs_size(self) -> int:
        return (
            1
            + 3
            * (
                self.num_subnets
                + self.num_subnets
                + self.num_subnets
                + self.obs_vector_hosts_per_subnet
                + self.obs_vector_hosts_per_subnet
            )
            + self.num_messages * self.message_length
        )

"""Agent-side CC4 evidence memory, shared by native evaluation and training.

File features consume only the acting Blue agent's returned Analyse observation.
The stock wrapper supplies Monitor flags and the public, padded host inventory.
No hidden compromise, session, or host-file state is inspected here.
"""

import numpy as np
from CybORG.Agents.Wrappers import BlueFlatWrapper, EnterpriseMAE
from gymnasium.spaces import Box

from jaxborg.blue_observation_contract import BLUE_HOST_SLOTS, CAGE4_ENHANCED_OBS_SIZE
from jaxborg.constants import BLUE_OBS_SIZE, NUM_SUBNETS, OBS_VECTOR_HOSTS_PER_SUBNET

SUBNET_BLOCK_SIZE = NUM_SUBNETS * 3 + OBS_VECTOR_HOSTS_PER_SUBNET * 2


class _EvidenceMemory:
    def __init__(self, env, *args, **kwargs):
        self._evidence = {}
        kwargs["pad_spaces"] = True
        super().__init__(env, *args, **kwargs)

    def reset(self, *args, **kwargs):
        self._evidence.clear()
        return super().reset(*args, **kwargs)

    def observation_space(self, agent_name):
        return Box(0.0, 2.0, shape=(CAGE4_ENHANCED_OBS_SIZE,), dtype=np.float32)

    def observation_change(self, agent_name, observation):
        base = np.asarray(super().observation_change(agent_name, observation), dtype=np.float32)
        if base.shape != (BLUE_OBS_SIZE,):
            raise ValueError(f"enhanced CC4 observations require the padded stock layout, got {base.shape}")
        # Rows: public host presence, last file evidence, process history, network history.
        memory = self._evidence.setdefault(agent_name, np.zeros((4, BLUE_HOST_SLOTS), dtype=np.float32))
        hosts = self.hosts(agent_name)
        slots = {}
        for subnet_slot, subnet in enumerate(self.subnets(agent_name)):
            subnet_hosts = [h for h in hosts if subnet in h and "router" not in h]
            if len(subnet_hosts) != OBS_VECTOR_HOSTS_PER_SUBNET:
                raise ValueError("enhanced CC4 observations require 16 padded host slots per subnet")
            start = subnet_slot * OBS_VECTOR_HOSTS_PER_SUBNET
            end = start + OBS_VECTOR_HOSTS_PER_SUBNET
            # BlueFixedActionWrapper.hosts includes padding; its action labels identify
            # real hosts using the scenario's public inventory, just as its action mask does.
            memory[0, start:end] = [h in self.env.environment_controller.state.hosts for h in subnet_hosts]
            offset = 1 + subnet_slot * SUBNET_BLOCK_SIZE + 3 * NUM_SUBNETS
            memory[2, start:end] = np.maximum(memory[2, start:end], base[offset : offset + 16])
            memory[3, start:end] = np.maximum(memory[3, start:end], base[offset + 16 : offset + 32])
            slots.update({h: start + i for i, h in enumerate(subnet_hosts)})

        action = observation.get("action")
        action_name = type(action).__name__
        target = getattr(action, "hostname", None)
        completed = getattr(observation.get("success"), "name", None) == "TRUE"
        if completed and target in slots:
            slot = slots[target]
            if action_name in ("Remove", "Restore"):
                # Recovery completion clears memory; transient flags remain in the prefix.
                memory[1:, slot] = 0
            elif action_name == "Analyse":
                files = observation.get(target, {}).get("Files", [])
                evidence = 0
                for file in files:
                    name = file.get("File Name", "")
                    # Native CC4 implants have density exactly 0.9, not > 0.9.
                    if file.get("Density", 0) >= 0.9 and not file.get("Signed", False):
                        evidence = max(evidence, 2 if name.startswith("escalate.") else 1)
                memory[1, slot] = evidence / 2.0
        memory[1:] *= memory[0]
        return np.concatenate([base, memory.reshape(-1)]).copy()


class EnhancedBlueFlatWrapper(_EvidenceMemory, BlueFlatWrapper):
    pass


class EnhancedEnterpriseMAE(_EvidenceMemory, EnterpriseMAE):
    pass


def enhanced_wrapper_class(wrapper_class):
    if wrapper_class is BlueFlatWrapper:
        return EnhancedBlueFlatWrapper
    if wrapper_class is EnterpriseMAE:
        return EnhancedEnterpriseMAE
    raise ValueError(f"cage4_enhanced_obs is unsupported for wrapper {wrapper_class}")

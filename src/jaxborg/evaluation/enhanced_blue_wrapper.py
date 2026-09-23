"""Agent-side CC4 evidence memory, shared by native evaluation and training.

File features consume only the acting Blue agent's returned Analyse observation.
The stock wrapper supplies Monitor flags and the public, padded host inventory.
No hidden compromise, session, or host-file state is inspected here.
"""

import numpy as np
from CybORG.Agents.Wrappers import BlueFlatWrapper, EnterpriseMAE
from gymnasium.spaces import Box

from jaxborg.blue_observation_contract import BLUE_HOST_SLOTS, blue_obs_size
from jaxborg.constants import BLUE_OBS_SIZE, NUM_SUBNETS, OBS_VECTOR_HOSTS_PER_SUBNET

SUBNET_BLOCK_SIZE = NUM_SUBNETS * 3 + OBS_VECTOR_HOSTS_PER_SUBNET * 2


class _EvidenceMemory:
    def __init__(self, env, *args, blue_observation_version=2, **kwargs):
        self.blue_observation_version = blue_observation_version
        self._evidence = {}
        self._ioc_pending = {}
        self._ioc_delivery = set()
        self._ioc_owners = None
        self._ioc_rank = {}
        self._ioc_tick = 0
        self._ioc_received = {}
        kwargs["pad_spaces"] = True
        super().__init__(env, *args, **kwargs)

    def reset(self, *args, **kwargs):
        self._evidence.clear()
        self._ioc_pending.clear()
        self._ioc_delivery.clear()
        self._ioc_owners = None
        self._ioc_rank.clear()
        self._ioc_received.clear()
        self._ioc_tick = 0
        return super().reset(*args, **kwargs)

    def observation_space(self, agent_name):
        return Box(0.0, 2.0, shape=(blue_obs_size(True, self.blue_observation_version),), dtype=np.float32)

    def _advance_ioc_tick(self):
        tick = self.env.environment_controller.step_count
        if self.blue_observation_version >= 2 and tick != self._ioc_tick:
            # Choose from the previous tick only; never expose a later agent's
            # observations to an earlier agent in the same conversion loop.
            self._ioc_delivery = set()
            for pending in self._ioc_pending.values():
                if pending:
                    host = min(pending, key=self._ioc_rank.__getitem__)
                    pending.remove(host)
                    self._ioc_delivery.add(host)
            self._ioc_tick = tick

    def _index_ioc_hosts(self):
        if self._ioc_owners is not None:
            return
        self._ioc_owners = {}
        inventory = self.env.environment_controller.state.hosts
        for agent in sorted(a for a in self.agents if "blue" in a):
            hosts = self.hosts(agent)
            for subnet_slot, subnet in enumerate(self.subnets(agent)):
                members = [h for h in hosts if subnet in h and "router" not in h]
                for slot, host in enumerate(members):
                    if host in inventory:
                        self._ioc_owners[host] = (agent, subnet_slot * 16 + slot)
                        self._ioc_rank[host] = (subnet, slot)
            self._ioc_pending[agent] = set()

    def _observe_decoy_sources(self, agent, host, memory):
        """Inspect only owned decoy ports and their visible connection events."""
        state = self.env.environment_controller.state
        if host not in state.hosts:
            return
        native_host = state.hosts[host]
        ports = {
            port["local_port"]
            for process in native_host.processes
            if getattr(process.decoy_type, "name", None) == "EXPLOIT"
            for port in process.open_ports
        }
        for event in native_host.events.old_network_connections + native_host.events.network_connections:
            if getattr(event, "local_port", None) not in ports:
                continue
            origin = state.ip_addresses.get(getattr(event, "remote_address", None))
            owner = self._ioc_owners.get(origin)
            if owner is None:
                continue
            receiver, slot = owner
            if receiver == agent:
                memory[4, slot] = 1.0
            else:
                self._ioc_pending[agent].add(origin)

    def observation_change(self, agent_name, observation):
        self._advance_ioc_tick()
        base = np.asarray(super().observation_change(agent_name, observation), dtype=np.float32)
        if base.shape != (BLUE_OBS_SIZE,):
            raise ValueError(f"enhanced CC4 observations require the padded stock layout, got {base.shape}")
        # Rows: public host presence, last file evidence, process history, network history.
        v2 = self.blue_observation_version >= 2
        memory = self._evidence.setdefault(agent_name, np.zeros((5 if v2 else 4, BLUE_HOST_SLOTS), dtype=np.float32))
        if v2:
            self._index_ioc_hosts()
            if self._ioc_received.get(agent_name) != self._ioc_tick:
                for host in self._ioc_delivery:
                    owner, slot = self._ioc_owners[host]
                    if owner == agent_name:
                        memory[4, slot] = 1.0
                self._ioc_received[agent_name] = self._ioc_tick
        hosts = self.hosts(agent_name)
        slots = {}
        current_alerts = np.zeros((2, BLUE_HOST_SLOTS), dtype=np.float32)
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
            current_alerts[:, start:end] = base[offset : offset + 32].reshape(2, 16)
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
                if v2:
                    # The H-MARL contract clears history, then adds currently
                    # visible alerts (including aged Monitor events).
                    memory[2:4, slot] = current_alerts[:, slot]
            elif action_name == "Analyse":
                files = observation.get(target, {}).get("Files", [])
                evidence = 0
                for file in files:
                    name = file.get("File Name", "")
                    # Native CC4 implants have density exactly 0.9, not > 0.9.
                    if file.get("Density", 0) >= 0.9 and not file.get("Signed", False):
                        evidence = max(evidence, 2 if name.startswith("escalate.") else 1)
                if evidence or not v2:
                    memory[1, slot] = evidence / 2.0
        memory[1:] *= memory[0]
        if not v2:
            return np.concatenate([base, memory.reshape(-1)]).copy()
        iocs = np.zeros(BLUE_HOST_SLOTS, dtype=np.float32)
        for subnet_slot, subnet in enumerate(self.subnets(agent_name)):
            members = [h for h in hosts if subnet in h and "router" not in h]
            for slot, host in enumerate(members, subnet_slot * 16):
                if not memory[0, slot]:
                    continue
                self._observe_decoy_sources(agent_name, host, memory)
                if memory[4, slot]:
                    iocs[slot] = 3
                    break
        # Match the shared JAX/H-MARL priority: files supersede decoy IOCs.
        iocs = np.where(memory[1] > 0, 3 - 2 * memory[1], iocs)
        return np.concatenate([base, memory[:4].reshape(-1), iocs / 3.0]).astype(np.float32)


class EnhancedBlueFlatWrapper(_EvidenceMemory, BlueFlatWrapper):
    pass


class EnhancedEnterpriseMAE(_EvidenceMemory, EnterpriseMAE):
    pass


def enhanced_wrapper_class(wrapper_class):
    if wrapper_class in (BlueFlatWrapper, EnhancedBlueFlatWrapper):
        return EnhancedBlueFlatWrapper
    if wrapper_class in (EnterpriseMAE, EnhancedEnterpriseMAE):
        return EnhancedEnterpriseMAE
    raise ValueError(f"cage4_enhanced_obs is unsupported for wrapper {wrapper_class}")

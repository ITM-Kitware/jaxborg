from jaxborg.state import SimulatorConst, SimulatorState


def apply_block_traffic(
    state: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
    src_subnet: int,
    dst_subnet: int,
) -> SimulatorState:
    blocked_zones = state.blocked_zones.at[dst_subnet, src_subnet].set(True)
    return state.replace(blocked_zones=blocked_zones)


def apply_allow_traffic(
    state: SimulatorState,
    const: SimulatorConst,
    agent_id: int,
    src_subnet: int,
    dst_subnet: int,
) -> SimulatorState:
    blocked_zones = state.blocked_zones.at[dst_subnet, src_subnet].set(False)
    return state.replace(blocked_zones=blocked_zones)

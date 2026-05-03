from jaxborg.state import SimulatorConst, SimulatorState


def apply_blue_analyse(state: SimulatorState, const: SimulatorConst, agent_id: int, target_host: int) -> SimulatorState:
    del const, agent_id, target_host
    # BlueFlatWrapper observations only expose monitor event state. Analyse
    # returns file artefacts in the raw CybORG observation, but those artefacts
    # are not projected into the flat observation vector used for training.
    return state

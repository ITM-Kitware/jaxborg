from jaxborg.state import CC4Const, CC4State


def apply_blue_analyse(state: CC4State, const: CC4Const, agent_id: int, target_host: int) -> CC4State:
    del const, agent_id, target_host
    # BlueFlatWrapper observations only expose monitor event state. Analyse
    # returns file artefacts in the raw CybORG observation, but those artefacts
    # are not projected into the flat observation vector used for training.
    return state

"""Native CybORG counterparts of the paper's JAX FSM selectors."""

from CybORG.Agents import FiniteStateRedAgent

from jaxborg.scenarios.cc4.hmarl_reds import HMARL_RED_TRANSFERS


class _HMARLRedAgent(FiniteStateRedAgent):
    profile: str

    def state_transitions_probability(self):
        matrix = super().state_transitions_probability()
        states, source, destination = HMARL_RED_TRANSFERS[self.profile]
        for state in states:
            matrix[state][destination] += matrix[state][source]
            matrix[state][source] = 0.0
        return matrix


class AggressiveRedAgent(_HMARLRedAgent):
    """Always uses aggressive service discovery (one tick)."""

    profile = "aggressive"


class StealthyRedAgent(_HMARLRedAgent):
    """Always uses stealth service discovery (three ticks)."""

    profile = "stealthy"


class ImpactRedAgent(_HMARLRedAgent):
    """Moves all Degrade Services probability to Impact."""

    profile = "impact"

from jaxborg.env import ScenarioEnv
from jaxborg.joint_env import JointPolicyCC4Env
from jaxborg.parity.fsm_red_env import FsmRedCC4Env


def test_scenario_env_preserves_domain_agent_names():
    env = ScenarioEnv(num_steps=1)

    expected = env.blue_agents + env.red_agents
    assert env.agents == expected
    assert set(env.action_spaces) == set(expected)
    assert set(env.observation_spaces) == set(expected)


def test_fsm_red_env_exposes_only_blue_agent_names():
    env = FsmRedCC4Env(num_steps=1)

    expected = list(env._env.blue_agents)
    assert env.agents == expected
    assert set(env.action_spaces) == set(expected)
    assert set(env.observation_spaces) == set(expected)


def test_joint_env_preserves_blue_and_red_agent_names():
    env = JointPolicyCC4Env(num_steps=1)

    expected = env.blue_agents + env.red_agents
    assert env.agents == expected
    assert set(env.action_spaces) == set(expected)
    assert set(env.observation_spaces) == set(expected)

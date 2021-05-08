from functools import partial

import pytest

from deeplookup.env import MalwareEnv
from deeplookup.experiments import pomdp


@pytest.mark.parametrize(
    "agent",
    [pomdp.dqn, pomdp.ddpg, pomdp.ddqn],
    ids=["dqn", "ddpg", "ddqn"],
)
def test_benchmark(agent, benchmark):
    env = MalwareEnv(log=False)

    function = partial(
        agent.test,
        env,
        nb_episodes=1,
        visualize=False,
        start_step_policy=env.start_step_policy,
        nb_max_start_steps=0,
        nb_max_episode_steps=20,
        verbose=0,
    )

    benchmark.pedantic(function, iterations=10, rounds=100)

from enum import Enum

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class Action(Enum):
    decrease_attention = 0
    increase_attention = 1
    access_detector = 2
    isolate_node = 3
    forget_node = 4


class State(Enum):
    healthy = 0
    infected = 1


class MalwareEnv(gym.Env):
    """
    Observations:
        Type: Box(2)
        Num    Observation           Min        Max
        0      Attention Level       0.05       1.0
        1      Malware Rate          0.0        1.0

    Actions:
        Type: Discrete(5)
        Num    Action
        0      Decrease attention
        1      Increase attention
        2      Access detector
        3      Isolate node
        4      Forget node

    Reward:
        Reward of -1 is awarded for accessing detector.
        Reward of -0.2 is awarded for decreasing attention.
        Reward of -0.8 is awarded for increasing attention.
        Reward of 1 is awarded for isolation of infected node.
        Reward of 1 is awarded for forgeting healthy node.
        Reward of -1 is awarded for isolation of healthy node.
        Reward of -1 if awarded for forgetting infected node.

    Starting State:
        Attention level is set between [0.1, 0.2]
        Actual state is set either to 'healthy' or 'infected'.

    Episode Termination:
        Node is either isolated of forgotten.
        Episode length is greater than 100.
    """

    def __init__(self, malware_prob: float = 0.9, seed: int = 100, log: bool = False):
        self.min_attention = 0.05
        self.max_attention = 1.0

        self.min_rate = 0.0
        self.max_rate = 1.0

        self.attention_inc = 0.05

        self.low = np.array([self.min_attention, self.min_rate], dtype=np.float32)
        self.high = np.array([self.max_attention, self.max_rate], dtype=np.float32)

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.malware_prob = malware_prob
        self.log = log

        # (attention, health)
        self.state = (None, None, None)
        self.latest_action = None
        self.actions = []
        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def start_step_policy(self, observation):
        attention, malware_rate = observation
        if attention > self.min_attention:
            return Action.access_detector.value
        return Action.increase_attention.value

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = np.argmax(action)

        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        action = Action(action)

        if self.log:
            self.actions.append(action)

        attention, malware_rate, health = self.state
        st = State(health)

        if action == Action.decrease_attention:
            attention = max(self.min_attention, attention - self.attention_inc)
        if action == Action.increase_attention:
            attention = min(self.max_attention, attention + self.attention_inc)
        if action == Action.access_detector:
            # Accessing a detector changes malware rate.
            #
            # When the node is healthy, there is a `1 - malware_prob` probability
            # to observe malware. And malware rate depends on the attention level.
            #
            # Throw a "dice" in order to calculate the malware rate.
            prob = self.np_random.uniform()
            T = (1 - self.malware_prob) if st == State.healthy else self.malware_prob

            mu = np.average([0, attention])
            # sigma = 0.2
            malware_rate = 0 if prob > T else self.np_random.normal(mu, 0.01)
            malware_rate = max(self.min_rate, malware_rate)
            malware_rate = min(self.max_rate, malware_rate)

        # Agent does not observe the node health directly, only through
        # malware rate.
        self.state = np.array([attention, malware_rate, health])
        self.latest_action = action

        observation = np.array([attention, malware_rate])
        reward = self.compute_reward(health, action)
        done = action in {Action.isolate_node, Action.forget_node}

        return observation, reward, done, {}  # {"state": self.state}

    def compute_reward(self, health, action):
        if action == Action.decrease_attention:
            return -0.2
        if action == Action.increase_attention:
            return -0.8
        if action == Action.access_detector:
            return -1
        if action == Action.isolate_node:
            return 1 * (health * 2 - 1)
        if action == Action.forget_node:
            return -1 * (health * 2 - 1)
        return 0

    def reset(self):
        # Node if either healthy (0), or infected (1), when node is infected,
        # agent observes malware requests depending on the attention level.
        health = self.np_random.choice([0, 1])
        attention = self.min_attention
        malware_rate = 0

        self.state = np.array([attention, malware_rate, health])
        return np.array([attention, malware_rate])

    def render(self, mode="human"):
        attention, malware_rate, infected = self.state
        print(f"\tattention: {attention} - malware rate: {malware_rate}", end=" - ")
        print(f"health: {'infected' if infected else 'healthy'}", end=" - ")
        print(f"action: {self.latest_action}")

    def close(self):
        pass

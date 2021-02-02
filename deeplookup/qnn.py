import pathlib
import pickle

import tensorflow as tf
from keras.callbacks import Callback
from keras.layers import Activation, Concatenate, Dense, Dropout, Flatten, Input
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from rl.random import OrnsteinUhlenbeckProcess


class Metrics(Callback):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.metrics = {}

    def on_train_begin(self, logs={}):
        self.metrics = {key: [] for key in self.agent.metrics_names}

    def on_step_end(self, episode_step, logs):
        for ordinal, key in enumerate(self.agent.metrics_names, 0):
            self.metrics[key].append(logs.get("metrics")[ordinal])


def fit_dqn(env, force: bool = False, dueling: bool = False, root_dir: str = ""):
    nb_actions = env.action_space.n

    loaded = False
    model_weights_path = pathlib.Path(f"{root_dir}/dqn{'-d' if dueling else ''}.h5")
    model_history_path = pathlib.Path(
        f"{root_dir}/dqn{'-d' if dueling else ''}.h5f.log"
    )

    if not force and model_weights_path.exists():
        model = load_model(str(model_weights_path))
        with open(model_history_path, "rb") as f:
            history = pickle.load(f)
        loaded = True
    else:
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Dense(64))
        model.add(Dropout(0.5))
        model.add(Activation("relu"))
        model.add(Dense(64))
        model.add(Dropout(0.5))
        model.add(Activation("relu"))
        model.add(Dense(64))
        model.add(Dropout(0.5))
        model.add(Activation("relu"))
        model.add(Dense(nb_actions))
        model.add(Activation("linear"))

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=1000,
        target_model_update=1e-2,
        policy=policy,
        enable_dueling_network=dueling,
        dueling_type="avg",
    )

    dqn.compile(Adam(lr=1e-3), metrics=["mae"])

    if loaded:
        return dqn, history

    metrics = Metrics(dqn)
    history = dqn.fit(
        env,
        nb_steps=10000,
        start_step_policy=env.start_step_policy,
        nb_max_start_steps=10,
        nb_max_episode_steps=100,
        callbacks=[metrics],
    )

    model.save(str(model_weights_path))
    with open(model_history_path, "wb") as f:
        history = history.history
        history.update(metrics.metrics)
        pickle.dump(history, f)

    return dqn, history


def fit_ddpg(env, force: bool = False, root_dir: str = ""):
    nb_actions = env.action_space.n

    loaded = False
    actor_weights_path = pathlib.Path(f"{root_dir}/ddpg-actor.h5")
    critic_weights_path = pathlib.Path(f"{root_dir}/ddpg-critic.h5")
    train_history_path = pathlib.Path(f"{root_dir}/ddpg.log")

    if not force and actor_weights_path.exists():
        actor = load_model(str(actor_weights_path))
        critic = load_model(str(critic_weights_path), compile=False)
        with open(train_history_path, "rb") as f:
            history = pickle.load(f)
        loaded = True
    else:
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        actor.add(Dense(16))
        actor.add(Dropout(0.5))
        actor.add(Activation("relu"))
        actor.add(Dense(16))
        actor.add(Dropout(0.5))
        actor.add(Activation("relu"))
        actor.add(Dense(16))
        actor.add(Dropout(0.5))
        actor.add(Activation("relu"))
        actor.add(Dense(nb_actions))
        actor.add(Activation("linear"))

        action_input = Input(shape=(nb_actions,), name="action_input")
        observation_input = Input(
            shape=(1,) + env.observation_space.shape, name="observation_input"
        )
        flattened_observation = Flatten()(observation_input)
        x = Concatenate()([action_input, flattened_observation])
        x = Dense(32)(x)
        x = Activation("relu")(x)
        x = Dense(32)(x)
        x = Activation("relu")(x)
        x = Dense(32)(x)
        x = Dropout(0.5)(x)
        x = Activation("relu")(x)
        x = Dense(1)(x)
        x = Activation("linear")(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)

    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(
        size=nb_actions, theta=0.15, mu=0.0, sigma=5
    )

    ddpg = DDPGAgent(
        nb_actions=nb_actions,
        actor=actor,
        critic=critic,
        critic_action_input=critic.inputs[0],
        memory=memory,
        nb_steps_warmup_critic=1000,
        nb_steps_warmup_actor=1000,
        random_process=random_process,
        gamma=0.99,
        target_model_update=1e-3,
    )

    ddpg.compile(Adam(lr=1e-3), metrics=["mae"])

    if loaded:
        return ddpg, history

    metrics = Metrics(ddpg)

    history = ddpg.fit(
        env,
        nb_steps=10000,
        start_step_policy=env.start_step_policy,
        nb_max_start_steps=10,
        nb_max_episode_steps=100,
        callbacks=[metrics],
    )

    actor.save(str(actor_weights_path))
    critic.save(str(critic_weights_path))
    with open(train_history_path, "wb") as f:
        history = history.history
        history.update(metrics.metrics)
        pickle.dump(history, f)

    return ddpg, history

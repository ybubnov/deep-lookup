import numpy as np

from deeplookup import qnn, vis
from deeplookup.env import MalwareEnv


ROOT_DIR = ".h5"


dqn, dqn_history = qnn.fit_dqn(MalwareEnv(), root_dir=ROOT_DIR)
ddqn, ddqn_history = qnn.fit_dqn(MalwareEnv(), dueling=True, root_dir=ROOT_DIR)
ddpg, ddpg_history = qnn.fit_ddpg(MalwareEnv(), root_dir=ROOT_DIR)


def main():
    for name, agent, history in [
        ("dqn", dqn, dqn_history),
        ("ddqn", ddqn, ddqn_history),
        ("ddpg", ddpg, ddpg_history),
    ]:
        env = MalwareEnv(log=True)

        test = agent.test(
            env,
            nb_episodes=1000,
            visualize=False,
            start_step_policy=env.start_step_policy,
            nb_max_start_steps=10,
            nb_max_episode_steps=100,
        )

        print(
            f"name: {name} - "
            f"avg: {np.average(test.history['episode_reward'])} - "
            f"min: {np.min(test.history['episode_reward'])} - "
            f"max: {np.max(test.history['episode_reward'])}"
        )

        p = vis.render_moving_average(
            history,
            "episode_reward",
            "Среднее вознаграждение за эпизод",
            "Вознаграждение",
        )
        p.figure.savefig(f"{ROOT_DIR}/images/{name}-reward.png", **vis.SAVE_KW)

        p = vis.render_moving_average(history, "loss", "Ошибка обучения", "Ошибка")
        p.figure.savefig(f"{ROOT_DIR}/images/{name}-loss.png", **vis.SAVE_KW)

        p = vis.render_moving_average(
            history, "mean_q", "Средняя ценность действий (Q)", "Ценность"
        )
        p.figure.savefig(f"{ROOT_DIR}/images/{name}-mean-q.png", **vis.SAVE_KW)

        p = vis.render_actions_histogram(env)
        p.figure.savefig(f"{ROOT_DIR}/images/{name}-hist.png", **vis.SAVE_KW)


if __name__ == "__main__":
    main()

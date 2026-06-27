import numpy as np

from teachai.games import CatcherGame, ImitationAgent, QLearningAgent


def _catch_rate(env, agent, episodes=200, explore=False):
    wins = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        info = {}
        while not done:
            action = (agent.act(state, explore=False)
                      if isinstance(agent, QLearningAgent) else agent.act(state))
            state, _, done, info = env.step(action)
        wins += int(info.get("caught", False))
    return wins / episodes


def test_demo_game_runs():
    env = CatcherGame(seed=0)
    state = env.reset()
    assert state.shape == (4,)
    state, reward, done, info = env.step(1)
    assert reward in (-1.0, 0.0, 1.0)
    assert isinstance(env.render(), str)


def test_qlearning_improves():
    env = CatcherGame(seed=0)
    agent = QLearningAgent(n_actions=env.n_actions, seed=0)
    agent.train(env, episodes=4000)
    rate = _catch_rate(CatcherGame(seed=123), agent)
    # A random agent catches ~1/width of the time; learned should be far better.
    assert rate > 0.8


def test_imitation_learns_from_expert():
    env = CatcherGame(seed=0)
    agent = ImitationAgent(n_actions=env.n_actions)
    agent.demonstrate(env, lambda e: e.expert_action(), episodes=300)
    agent.learn(epochs=50)
    rate = _catch_rate(CatcherGame(seed=7), agent)
    assert rate > 0.8


def test_qagent_persistence(tmp_path):
    env = CatcherGame(seed=0)
    agent = QLearningAgent(n_actions=env.n_actions, seed=0)
    agent.train(env, episodes=1500)
    path = tmp_path / "q.json"
    agent.save(path)
    reloaded = QLearningAgent.load(path)
    base = _catch_rate(CatcherGame(seed=5), agent)
    same = _catch_rate(CatcherGame(seed=5), reloaded)
    assert abs(base - same) < 1e-9

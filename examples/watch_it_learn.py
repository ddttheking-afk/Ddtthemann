"""Watch the AI learn the demo game from scratch, then play a visible round.

Run:  python examples/watch_it_learn.py
"""

from teachai.games import CatcherGame, QLearningAgent


def main() -> None:
    env = CatcherGame(seed=0)
    agent = QLearningAgent(n_actions=env.n_actions, seed=0)

    print("Teaching the AI by reinforcement (trial and reward)...")
    history = agent.train(env, episodes=4000)
    start = (sum(history[:200]) / 200 + 1) / 2
    end = (sum(history[-200:]) / 200 + 1) / 2
    print(f"  catch-rate went {start:.0%}  ->  {end:.0%}\n")

    print("Now playing one round you can watch:\n")
    state = env.reset()
    done = False
    info = {}
    while not done:
        print(env.render())
        print("-" * env.width)
        state, _, done, info = env.step(agent.act(state))
    print("Result:", "CAUGHT IT!" if info.get("caught") else "missed")


if __name__ == "__main__":
    main()

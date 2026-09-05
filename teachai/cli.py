"""Command-line interface for TeachAI.

Run ``python -m teachai --help`` to get started.

Examples
--------
Teach it pictures, then use it::

    python -m teachai vision teach --data photos/ --model brain.npz
    python -m teachai vision classify --model brain.npz --image mystery.jpg
    python -m teachai vision select  --model brain.npz --target cat pile/*.jpg

Watch it learn the built-in game, then play it::

    python -m teachai game train --episodes 3000 --model catcher.json
    python -m teachai game play  --model catcher.json --episodes 5 --render
    python -m teachai game imitate --episodes 5 --render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# vision
# --------------------------------------------------------------------------- #
def _vision_teach(args) -> None:
    from .vision import ImageSelector

    sel = ImageSelector()
    sel.teach_from_folder(args.data, epochs=args.epochs)
    sel.save(args.model)
    print(f"Taught {len(sel.model.classes_)} categories: "
          f"{', '.join(map(str, sel.model.classes_))}")
    print(f"Saved brain to {args.model}")


def _vision_classify(args) -> None:
    from .vision import ImageSelector

    sel = ImageSelector.load(args.model)
    label, conf = sel.classify(args.image)
    print(f"{args.image}: {label}  (confidence {conf:.0%})")


def _vision_select(args) -> None:
    from .vision import ImageSelector

    sel = ImageSelector.load(args.model)
    ranked = sel.rank(args.images, args.target)
    print(f"Best match for '{args.target}':\n  {ranked[0][0]}  "
          f"({ranked[0][1]:.0%})")
    if args.verbose:
        print("Full ranking:")
        for path, score in ranked:
            print(f"  {score:6.1%}  {path}")


# --------------------------------------------------------------------------- #
# game
# --------------------------------------------------------------------------- #
def _game_train(args) -> None:
    from .games import CatcherGame, QLearningAgent

    env = CatcherGame(seed=args.seed)
    agent = QLearningAgent(n_actions=env.n_actions, seed=args.seed)
    history = agent.train(env, episodes=args.episodes)
    window = max(1, args.episodes // 20)
    early = sum(history[:window]) / window
    late = sum(history[-window:]) / window
    print(f"Trained {args.episodes} episodes.")
    print(f"  catch-rate at start: {(early + 1) / 2:.0%}")
    print(f"  catch-rate at end:   {(late + 1) / 2:.0%}")
    agent.save(args.model)
    print(f"Saved agent to {args.model}")


def _game_play(args) -> None:
    from .games import CatcherGame, QLearningAgent

    env = CatcherGame(seed=args.seed)
    agent = QLearningAgent.load(args.model)
    wins = 0
    for ep in range(args.episodes):
        state = env.reset()
        done = False
        while not done:
            if args.render:
                print(env.render())
                print("-" * env.width)
            action = agent.act(state, explore=False)
            state, reward, done, info = env.step(action)
        wins += int(info.get("caught", False))
        if args.render:
            print(f"Episode {ep + 1}: {'CAUGHT' if info.get('caught') else 'missed'}\n")
    print(f"Caught {wins}/{args.episodes}")


def _game_imitate(args) -> None:
    from .games import CatcherGame, ImitationAgent

    env = CatcherGame(seed=args.seed)
    agent = ImitationAgent(n_actions=env.n_actions)
    # Use the game's built-in expert as the "human" demonstrator.
    agent.demonstrate(env, lambda e: e.expert_action(), episodes=200)
    agent.learn(epochs=40)
    wins = 0
    for ep in range(args.episodes):
        state = env.reset()
        done = False
        while not done:
            if args.render:
                print(env.render())
                print("-" * env.width)
            action = agent.act(state)
            state, reward, done, info = env.step(action)
        wins += int(info.get("caught", False))
    print(f"Imitation agent caught {wins}/{args.episodes} after watching a demo.")


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="teachai",
        description="Teach one AI to select pictures and play games.",
    )
    sub = p.add_subparsers(dest="group", required=True)

    # vision -------------------------------------------------------------- #
    vision = sub.add_parser("vision", help="teach and use the picture brain")
    vsub = vision.add_subparsers(dest="cmd", required=True)

    vt = vsub.add_parser("teach", help="teach from a folder of category subfolders")
    vt.add_argument("--data", required=True, help="folder whose subfolders are categories")
    vt.add_argument("--model", default="vision.npz", help="where to save the brain")
    vt.add_argument("--epochs", type=int, default=25)
    vt.set_defaults(func=_vision_teach)

    vc = vsub.add_parser("classify", help="classify one image")
    vc.add_argument("--model", required=True)
    vc.add_argument("--image", required=True)
    vc.set_defaults(func=_vision_classify)

    vs = vsub.add_parser("select", help="pick the best image for a target category")
    vs.add_argument("--model", required=True)
    vs.add_argument("--target", required=True, help="category to select for")
    vs.add_argument("images", nargs="+", help="candidate image paths")
    vs.add_argument("--verbose", "-v", action="store_true")
    vs.set_defaults(func=_vision_select)

    # game ---------------------------------------------------------------- #
    game = sub.add_parser("game", help="teach and run the game agent")
    gsub = game.add_subparsers(dest="cmd", required=True)

    gt = gsub.add_parser("train", help="learn the demo game by reinforcement")
    gt.add_argument("--episodes", type=int, default=3000)
    gt.add_argument("--model", default="catcher.json")
    gt.add_argument("--seed", type=int, default=0)
    gt.set_defaults(func=_game_train)

    gp = gsub.add_parser("play", help="play the demo game with a trained agent")
    gp.add_argument("--model", required=True)
    gp.add_argument("--episodes", type=int, default=5)
    gp.add_argument("--render", action="store_true")
    gp.add_argument("--seed", type=int, default=1)
    gp.set_defaults(func=_game_play)

    gi = gsub.add_parser("imitate", help="learn the demo game by copying a demo")
    gi.add_argument("--episodes", type=int, default=5)
    gi.add_argument("--render", action="store_true")
    gi.add_argument("--seed", type=int, default=1)
    gi.set_defaults(func=_game_imitate)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

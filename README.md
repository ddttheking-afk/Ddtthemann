# TeachAI

A small **AI you teach by example**. One learning model — one "brain" — that you
keep teaching over time to do two kinds of things:

1. **Select & recognise pictures** — show it example images grouped by what they
   are, then it can classify a new picture or *pick the best one* for a target.
2. **Play games** — it learns a built-in game you can watch right now, and the
   same agents can drive a **real emulator** (e.g. **Pokémon** in mGBA, or
   **Epic Seven** in an Android emulator) by watching the screen and pressing keys.

It runs on plain Python + numpy, so you can get it going on a normal laptop.

> **What this is honest about:** TeachAI is a *teachable foundation*, not a
> pre-trained super-bot. It gets good at things **because you teach it** —
> either by showing it labelled pictures, by letting it play and rewarding it,
> or by playing yourself while it copies you. The demo game proves the whole
> loop works end-to-end in seconds; real emulators need you to wire up the game
> window and put in some teaching time.

---

## Install

```bash
git clone https://github.com/ddttheking-afk/ddtthemann.git
cd ddtthemann
pip install -e .            # core + vision
# optional, only for real emulators:
pip install -e ".[emulator]"
```

(You can also just `pip install -r requirements.txt` and run with
`PYTHONPATH=.`)

---

## 1. Teach it pictures

Put example images in folders named after each category:

```
photos/
  cat/   img1.jpg img2.jpg ...
  dog/   img1.jpg img2.jpg ...
```

```bash
teachai vision teach   --data photos/ --model brain.npz
teachai vision classify --model brain.npz --image mystery.jpg
#   mystery.jpg: cat  (confidence 93%)
teachai vision select  --model brain.npz --target cat pile/*.jpg
#   Best match for 'cat':  pile/photo7.jpg  (88%)
```

Or in Python:

```python
from teachai.vision import ImageSelector

sel = ImageSelector()
sel.teach_from_folder("photos")          # learn the categories
sel.classify("mystery.jpg")              # -> ("cat", 0.93)
sel.select_best(glob("pile/*.jpg"), "cat")   # -> best cat picture
```

Try it with zero setup: `python examples/teach_pictures.py`.

---

## 2. Watch it learn a game

The built-in **Catcher** game lets you see learning happen immediately:

```bash
teachai game train --episodes 3000 --model catcher.json
#   catch-rate at start: 21%
#   catch-rate at end:   99%
teachai game play  --model catcher.json --episodes 5 --render
```

Or the "copy me" approach (imitation learning):

```bash
teachai game imitate --episodes 5 --render
```

Try it: `python examples/watch_it_learn.py`.

There are **two ways it learns**, and you pick whichever suits a game:

| Way | How you teach it | Best when |
|-----|------------------|-----------|
| **Imitation** (`ImitationAgent`) | You play; it copies your moves | You can play the game yourself |
| **Reinforcement** (`QLearningAgent`) | It plays; you supply a reward | You can score win/lose from the screen |

---

## 3. Point it at a real emulator (Pokémon, Epic Seven, …)

The same agents drive any on-screen game through the **emulator adapter**, which
captures a region of your screen as the "state" and sends key presses as
"actions". See **[docs/EMULATORS.md](docs/EMULATORS.md)** for the full walkthrough.

The short version (imitation — no reward function needed):

```python
from teachai.games.emulator import EmulatorEnv, record_human_play
from teachai.games import ImitationAgent

# The rectangle on screen where the game is, and which key = which action.
env = EmulatorEnv(region=(60, 120, 480, 320),
                  keymap=["left", "right", "up", "down", "x", "z"])

# Play the game yourself; it records what you do.
samples = record_human_play(env, action_for_key={
    "a": 0, "d": 1, "w": 2, "s": 3, "x": 4, "z": 5,
})

agent = ImitationAgent(n_actions=env.n_actions)
for state, action in samples:
    agent.record(state, action)
agent.learn()

# Now let it play.
state = env.reset()
for _ in range(1000):
    state, reward, done, _ = env.step(agent.act(state))
```

---

## How it all fits together

```
teachai/
  core/      the shared brain
    model.py      TeachableModel  -- online classifier you keep teaching
    features.py   turn images / screen frames into vectors
  vision/
    selector.py   ImageSelector   -- teach, classify, select pictures
  games/
    env.py        Environment     -- the common game interface
    demo_game.py  CatcherGame     -- built-in game to watch it learn
    agent.py      ImitationAgent + QLearningAgent
    emulator.py   EmulatorEnv     -- drive real emulators via screen + keys
    matchers.py   TemplateReward  -- turn a win-screen screenshot into a reward
  cli.py          the `teachai` command
```

Everything — pictures and games — is powered by the one `TeachableModel`, so as
you teach it more, you're growing a single brain.

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest -q
```

## A note on emulators & ROMs

TeachAI controls software that's already running on your own machine; it doesn't
download games or ROMs and doesn't bundle any. Only use it with games you own
and in line with each game's terms of service.

## License

MIT — see [LICENSE](LICENSE).

# Driving real emulators with TeachAI

This guide shows how to point TeachAI at a real game running on your computer —
for example **Pokémon** in an emulator like [mGBA](https://mgba.io/), or
**Epic Seven** in an Android emulator such as BlueStacks/LDPlayer (or a phone
mirrored to your PC with [scrcpy](https://github.com/Genymobile/scrcpy)).

The idea is simple: TeachAI **looks at a rectangle of your screen** (the game
window) and **presses keys** to act, exactly the way you do. It learns either by
**copying you** (imitation) or by **trial and reward** (reinforcement).

> Only use this with games you own, and within each game's terms of service.
> Many online games forbid automation — check before you run a bot.

---

## 1. Install the extra dependencies

Screen capture and key sending are platform-specific, so they're optional:

```bash
pip install -e ".[emulator]"      # installs mss + pynput
```

---

## 2. Find the game's screen rectangle

`EmulatorEnv` needs `region=(left, top, width, height)` in screen pixels — the
box around the game's play area. Easiest way: maximise the emulator, take a
screenshot, and read off the coordinates, or start with a rough guess and adjust.

```python
from teachai.games.emulator import EmulatorEnv

env = EmulatorEnv(
    region=(60, 120, 480, 320),       # <-- your game window
    keymap=["left", "right", "up", "down", "x", "z"],
)
print(env.render())                   # confirms it can grab the frame
```

`keymap` maps **action index → keyboard key**. Use the same keys your emulator is
configured to use (in mGBA, Pokémon's A/B default to X/Z, D-pad to arrows).
`None` in the list means "do nothing" for that action.

---

## 3a. Teach it by playing (imitation — recommended first)

No reward function needed. You play; it records `(screen, your key)` pairs and
learns to reproduce them.

```python
from teachai.games.emulator import EmulatorEnv, record_human_play
from teachai.games import ImitationAgent

env = EmulatorEnv(region=(60, 120, 480, 320),
                  keymap=["left", "right", "up", "down", "x", "z"])

# Map the keys you'll press while demonstrating to action indices.
samples = record_human_play(env, action_for_key={
    "a": 0, "d": 1, "w": 2, "s": 3, "x": 4, "z": 5,
})   # play now; press Esc to stop recording

agent = ImitationAgent(n_actions=env.n_actions)
for state, action in samples:
    agent.record(state, action)
agent.learn()
agent.save("pokemon_agent.npz")

# Let it play what it learned.
state = env.reset()
for _ in range(2000):
    state, _, done, _ = env.step(agent.act(state))
    if done:
        state = env.reset()
```

The more varied your demonstrations, the better it generalises.

---

## 3b. Teach it by reward (reinforcement)

Use this when the agent can play unattended and you can *score* the screen —
e.g. detect a "Victory" banner in Epic Seven, or a win screen in Pokémon. The
easiest reliable signal is: **save a screenshot of the winning moment**, and let
`TemplateReward` recognise it for you — no pixel math to write:

```python
from teachai.games.emulator import EmulatorEnv
from teachai.games import QLearningAgent, TemplateReward

# 1. Screenshot the "Victory" banner once and crop it to victory.png.
# 2. Turn that screenshot into a reward + episode-end detector.
win = TemplateReward("victory.png", threshold=0.7)

env = EmulatorEnv(
    region=(60, 120, 480, 320),
    keymap=["x", "z", None],           # attack, confirm, wait
    reward_fn=win,                     # +1 when the banner appears
    done_fn=win.detected,              # episode ends on the win screen
    step_delay=0.3,
)

agent = QLearningAgent(n_actions=env.n_actions)
agent.train(env, episodes=500)         # this plays the real game live
agent.save("e7_agent.json")
```

`TemplateReward` compares a small, brightness-tolerant grayscale version of the
screen against your saved image, so it survives minor animation flicker. Tips:

- **Crop tightly** to just the banner and pass `region=(left, top, w, h)` (the
  area of the frame where it appears) — narrower is far more reliable.
- **Tune `threshold`** (0..1): raise it if it triggers too easily, lower it if it
  misses. Use `win.score(frame)` to see the raw similarity while calibrating.

If you'd rather hand-roll the signal, any `frame -> float` function works as
`reward_fn` and any `frame -> bool` as `done_fn`.

---

## Tips & gotchas

- **Start with the demo game.** `teachai game train` proves your install and the
  learning loop work before you fight emulator quirks.
- **Slow the loop down.** Real games need time to animate — raise `step_delay`
  until actions actually register.
- **Keep the game window in the same place.** The `region` is fixed pixel
  coordinates; if you move the window, re-measure it.
- **Focus matters.** Key presses go to whatever window is focused — keep the
  emulator focused while the agent runs.
- **Epic Seven specifics.** It's a phone game, so run it in an Android emulator
  or mirror your phone with scrcpy, then treat that window like any other. Its
  auto-battle is turn-based, which suits reward-based learning on the result
  screen.
- **Pokémon specifics.** Map the D-pad to arrow keys and A/B to your emulator's
  configured keys; imitation learning is a great fit because you can simply play
  through routes while it watches.

---

## How the adapter works (under the hood)

`EmulatorEnv` implements the same `Environment` interface as the demo game:

- `reset()` / `step(action)` grab a frame with **mss**, shrink it to a small
  normalised vector with `teachai.core.features.frame_to_vector`, and (for
  `step`) send a key with **pynput**.
- Because the state is "just a vector", the *exact same* `ImitationAgent` and
  `QLearningAgent` that learned the demo game work here unchanged.

That shared interface is the whole point: learn the pattern on the toy game,
apply it to the real one.

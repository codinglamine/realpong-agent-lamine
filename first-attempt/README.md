# First attempt — policy gradient, and why it was not enough

This folder is the approach I tried before the CNN in the root of this repository.
It did not work. It is here because the reasons it did not work are what produced
the version that did.

## The task

The tournament ships a trained opponent, `realpong` — a pixel policy-gradient
network — and a battle template. You write an agent that reads the raw Atari Pong
frame `(210, 160, 3)` and returns UP or DOWN, and you have to beat it.

## What I built first

The obvious move: build the same class of thing as the opponent, following
Karpathy's *Pong from pixels*.

| Component | Choice |
|---|---|
| Input | 80x80 binarised frame, minus the previous frame, flattened to 6400 |
| Network | one hidden layer, 200 units, sigmoid head giving P(UP) |
| Objective | REINFORCE, discount reset at every point boundary |
| Variance reduction | value head as a baseline, advantage normalisation |
| Optimiser | Adam, gradient accumulation over 10 episodes |

`agent_ale.py` holds the network, the preprocessing and the agent contract.
`realpong.py` trains it against a scripted ball-following opponent with a
curriculum. `train_realpong_ale.py` trains it on single-agent `ALE/Pong-v5`.

## Three things went wrong

**1. The selection criterion was measuring the wrong thing.**

The first version saved its "best" checkpoint by shaped reward. What that
selected was an agent that followed the ball beautifully and never scored a
point — shaped reward rewarded being in the right place, not winning. Every
checkpoint I had saved was the same useless policy.

The fix is in `train_realpong_ale.py`: a checkpoint is only promoted if it
improves the **win rate over the last 100 games**, tie-broken on running reward.
A win is a point differential above zero and nothing else.

**2. The environment would not build.**

`pong_v3` needs `multi_agent_ale_py`, which has no wheel for Python 3.14 on
Windows and needs a Visual C++ toolchain to build from source. Rather than fight
it, I trained on single-agent `ALE/Pong-v5`, which serves the same 210x160x3
Atari frames. Because the preprocessing is identical, weights trained there load
straight into the tournament agent without changes.

**3. The network was too small and the budget was too long.**

`realpong_ale_log.csv` is the whole run. 125 episodes:

| | |
|---|---|
| Mean score, first 50 games | -20.4 |
| Mean score, last 50 games | -19.8 |
| Best single game | -16, so 21-5 |
| Games won | 0 of 125 |

It was learning — the trend is real, not noise — but 200 hidden units learning
from scratch on a sparse win/loss signal needed thousands of episodes on a
laptop CPU. The tournament deadline was days away.

## What I did instead

The three findings pointed at the same conclusion: the problem was not tuning,
it was capacity and where the learning signal comes from.

1. **Match the opponent's capacity.** The strong agent was roughly 1.7M
   parameters. A 200-unit hidden layer cannot represent that policy, so no amount
   of training time gets there.
2. **Do not learn the basics from scratch.** Distil the strong agent first — get
   to its level by imitation in a fraction of the time, then start improving.
3. **Improve with self-play.** PPO against a pool of past selves, with a pure
   win/loss reward and no shaping at all, which is the mistake from finding 1
   turned into a rule.
4. **Keep the gated promotion.** A checkpoint only becomes "best" if it beats the
   target on wins first, net score second.

That is the agent in the root of this repository, and it beat the opponent.

## Running this

```bash
pip install numpy torch gymnasium ale-py
python train_realpong_ale.py      # resumes from realpong_ale.pt, Ctrl-C to stop
```

`realpong_ale_out.txt` is the unedited console output of the run above.

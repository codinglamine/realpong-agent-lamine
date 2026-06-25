# RealPong agent, Lamine

Two trained CNN agents, one per arena. Each reads the 80x80 frame (2 channels: position + motion)
and outputs UP/DOWN every step. Both files are self-contained (only `numpy` + `torch`); the arena
imports the module and calls `Agent(weights_path)`.

## What to submit

| Arena   | Run as                              | Network            |
|---------|-------------------------------------|--------------------|
| Regular | `lamine.py:lamine.pt`               | ~1.7M-param CNN     |
| Chaos   | `lamine_chaos.py:lamine_chaos.pt`   | CNN                |

```
regular:  lamine.py:lamine.pt
chaos:    lamine_chaos.py:lamine_chaos.pt
```

## How it was built

- Matched the strong opponent's capacity (~1.7M params) so the network could represent its policy.
- Cloned (distilled) the strong agent to reach its level fast, then ran PPO self-play against a pool
  of past selves with a pure win/loss reward to push past it. One clean technique, no shaping.
- Gated promotion: a checkpoint only becomes "best" if it beats the target on (wins, then net score).

## Run it locally

Drop these next to `arena.py` / `arena_chaos.py` from the tournament repo:

```
python arena.py       lamine.py:lamine.pt             willem-cnn.py:willem-cnn.pt --best-of 3
python arena_chaos.py lamine_chaos.py:lamine_chaos.pt willem-cnn.py:willem-cnn.pt --best-of 3
```

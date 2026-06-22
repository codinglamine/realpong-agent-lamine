# realpong-agent-lamine

My agent for the [RealPong tournament arena](https://github.com/Helmus101/tournament) — an 80×80 symmetric Pong where two paddles rally a ball that **accelerates on every hit**, until one reaches 21.

## Files

| file | what it is |
|------|------------|
| `newfolder.py` | the model (a small CNN) + the `Agent` class the arena calls |
| `newfolder_trained_best.pt` | the trained weights |
| `SUBMIT.txt` | submission notes |

Self-contained — only `numpy` + `torch`, no local imports.

## Submit as

```
newfolder.py:newfolder_trained_best.pt
```

## How the arena uses it

It imports `newfolder.py`, builds `Agent("newfolder_trained_best.pt")`, then calls
`agent.reset()` at the start of each game and `agent.act(frame)` every step.
`frame` is an 80×80 binary image with **your paddle on the right**; `act()` returns `2` (UP) or `3` (DOWN).

## Test locally

```bash
python arena.py newfolder.py:newfolder_trained_best.pt realpong.py:realpong.pt bf
```

## Results

Current "new ball physics" arena, round-robin, best of 3 to 21:

| opponent | result |
|----------|--------|
| repo `realpong` | **21–6, 21–2** (set won) |
| `bf` (tracker) | **21–0, 21–0** (0 conceded) |
| round-robin | **🏆 CHAMPION** |

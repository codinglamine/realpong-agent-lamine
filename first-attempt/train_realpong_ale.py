"""
train_realpong_ale.py -- train the ARENA's Karpathy PolicyNet on single-agent ALE/Pong-v5
and save an arena-compatible realpong.pt.

WHY single-agent ALE/Pong-v5 (not pong_v3):
  pong_v3 needs multi_agent_ale_py, which won't build on this Windows box (no wheel for
  py3.14, source build needs VC++). Single-agent ale_py 0.12 works locally. The frames are
  the SAME Atari Pong frames (210x160x3); agent_ale.preprocess binarizes them, so a net
  trained here loads directly into agent_ale.Agent / run_tournament.py. The agent is the
  RIGHT paddle = first_0; the arena mirrors the left leg, so right-paddle training covers both.

ALGORITHM (matches realpong.py):
  Karpathy policy gradient, P(UP) head, frame-difference input, per-point discount reset,
  value-head baseline -> advantage, advantage normalization, Adam, batched updates.

"DO THE BEST" FIX:
  Saves the BEST checkpoint by actual WIN RATE over the last 100 games (tie-break: running
  reward) -- NOT by shaped reward. This is the fix for the New-folder bug where "best" saved a
  ball-tracker that never scored. Win = episode point differential > 0.

OUTPUTS (in this folder):
  realpong_best.pt   {"model": state_dict}  <- submit this (copy to realpong.pt for the arena)
  realpong_ale.pt    full checkpoint (model+optimizer+counters) for resume
  realpong_ale_log.csv

Run:  python train_realpong_ale.py            (resumes realpong_ale.pt; Ctrl-C to stop)
"""
import os
import csv
import numpy as np
import torch
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

from agent_ale import PolicyNet, preprocess, UP, DOWN, D, DEVICE

# ── hyperparameters (Karpathy core + realpong.py stack) ────────────────────────
LEARNING_RATE = 1e-3
GAMMA         = 0.99
VALUE_COEF    = 0.5
BATCH_SIZE    = 10        # episodes per optimizer step (grad accumulation)
WIN_WINDOW    = 100       # win rate measured over last 100 games
SAVE_EVERY    = 25        # episodes between resume-checkpoint writes
SEED          = 1

HERE      = os.path.dirname(os.path.abspath(__file__))
BEST_PATH = os.path.join(HERE, "realpong_best.pt")   # arena submission
CKPT_PATH = os.path.join(HERE, "realpong_ale.pt")    # resume
LOG_PATH  = os.path.join(HERE, "realpong_ale_log.csv")


def discount_rewards(r):
    out = np.zeros_like(r, dtype=np.float64)
    add = 0.0
    for t in reversed(range(r.size)):
        if r[t] != 0:
            add = 0.0                        # reset at each point (Pong-specific)
        add = add * GAMMA + r[t]
        out[t] = add
    return out


def main():
    torch.manual_seed(SEED)
    net = PolicyNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    ep          = 0
    running     = None
    best_wr     = -1.0
    best_run    = float("-inf")
    recent_wins = []                          # last WIN_WINDOW outcomes (1/0)

    if os.path.exists(CKPT_PATH):
        ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
        net.load_state_dict(ck["model"])
        try: opt.load_state_dict(ck["optimizer"])
        except Exception: pass
        ep          = ck.get("episode", 0)
        running     = ck.get("running", None)
        best_wr     = ck.get("best_wr", -1.0)
        best_run    = ck.get("best_run", float("-inf"))
        recent_wins = ck.get("recent_wins", [])
        print(f"resumed {os.path.basename(CKPT_PATH)} at ep {ep} | running {running} | best_wr {best_wr:.1f}%")
    else:
        print("fresh PolicyNet")

    if ep == 0 or not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["episode", "reward", "running", "wr100"])

    # ALE/Pong-v5: agent = right paddle, built-in CPU = left. No sticky actions (match arena).
    env = gym.make("ALE/Pong-v5", repeat_action_probability=0.0, frameskip=4)

    print(f"training on ALE/Pong-v5 (RIGHT paddle) | lr={LEARNING_RATE} | best-by-WIN-RATE | Ctrl-C to stop\n")
    opt.zero_grad()
    try:
        while True:
            obs, _ = env.reset(seed=SEED + ep)
            prev = None
            logps, values, rewards = [], [], []
            done = False

            while not done:
                cur  = preprocess(obs)
                diff = cur - prev if prev is not None else np.zeros(D, np.float32)
                prev = cur

                prob, value = net(torch.from_numpy(diff).unsqueeze(0).to(DEVICE))
                prob = prob.squeeze(0)
                up   = torch.rand((), device=DEVICE) < prob
                action = UP if up.item() else DOWN
                logps.append(torch.log((prob if up else 1 - prob) + 1e-8))
                values.append(value.squeeze(0))

                obs, r, term, trunc, _ = env.step(action)
                rewards.append(float(r))
                done = term or trunc

            if not rewards:                    # degenerate episode, skip
                continue

            # ── policy gradient with value baseline ──
            returns  = torch.tensor(discount_rewards(np.array(rewards)),
                                    dtype=torch.float32, device=DEVICE)
            values_t = torch.stack(values)
            adv      = returns - values_t.detach()
            adv      = (adv - adv.mean()) / (adv.std() + 1e-8)
            policy_loss = -(torch.stack(logps) * adv).sum()
            value_loss  = VALUE_COEF * (values_t - returns).pow(2).mean()
            (policy_loss + value_loss).backward()

            ep += 1
            if ep % BATCH_SIZE == 0:
                opt.step(); opt.zero_grad()

            reward_sum = float(sum(rewards))
            won        = 1 if reward_sum > 0 else 0
            recent_wins.append(won)
            if len(recent_wins) > WIN_WINDOW:
                recent_wins.pop(0)
            wr = sum(recent_wins) / len(recent_wins) * 100
            running = reward_sum if running is None else running * 0.99 + reward_sum * 0.01

            print(f"ep {ep:5d} | reward {reward_sum:+3.0f} | running {running:+6.2f} | "
                  f"wr{WIN_WINDOW} {wr:5.1f}% ({sum(recent_wins)}/{len(recent_wins)})", flush=True)
            with open(LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow([ep, f"{reward_sum:.0f}", f"{running:.4f}", f"{wr:.2f}"])

            # ── best-by-WIN-RATE (tie-break: running reward); needs a full window ──
            if len(recent_wins) >= WIN_WINDOW and (wr > best_wr or (wr == best_wr and running > best_run)):
                best_wr, best_run = wr, running
                torch.save({"model": net.state_dict()}, BEST_PATH)
                print(f"  *** NEW BEST  wr={wr:.1f}%  running={running:+.2f}  -> {os.path.basename(BEST_PATH)} ***")

            if ep % SAVE_EVERY == 0:
                torch.save({
                    "model": net.state_dict(), "optimizer": opt.state_dict(),
                    "episode": ep, "running": running,
                    "best_wr": best_wr, "best_run": best_run,
                    "recent_wins": recent_wins,
                }, CKPT_PATH)

    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        torch.save({
            "model": net.state_dict(), "optimizer": opt.state_dict(),
            "episode": ep, "running": running,
            "best_wr": best_wr, "best_run": best_run,
            "recent_wins": recent_wins,
        }, CKPT_PATH)
        env.close()
        print(f"saved {os.path.basename(CKPT_PATH)} (ep {ep})")


if __name__ == "__main__":
    main()

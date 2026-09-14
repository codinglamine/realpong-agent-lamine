"""
promote_to_realpong.py -- copy the best-by-win-rate checkpoint into the arena filename.

The arena (run_tournament.py) loads `realpong.pt`. Training saves the best model to
`realpong_best.pt`. Run this when you're happy with training to make it the submission:

    python promote_to_realpong.py

It backs up any existing realpong.pt to realpong.prev.pt first, then verifies the
promoted file loads into agent_ale.Agent.
"""
import os, shutil, torch
from agent_ale import Agent, PolicyNet

HERE = os.path.dirname(os.path.abspath(__file__))
BEST = os.path.join(HERE, "realpong_best.pt")
DEST = os.path.join(HERE, "realpong.pt")

if not os.path.exists(BEST):
    raise SystemExit(f"no {BEST} yet -- let training run until it prints 'NEW BEST'.")

if os.path.exists(DEST):
    shutil.copy2(DEST, os.path.join(HERE, "realpong.prev.pt"))
    print("backed up existing realpong.pt -> realpong.prev.pt")

# normalize to a clean {"model": state_dict} the arena expects
ck = torch.load(BEST, map_location="cpu", weights_only=False)
state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
torch.save({"model": state}, DEST)

# verify it loads through the exact arena interface
Agent(DEST)
print(f"promoted {os.path.basename(BEST)} -> realpong.pt  (loads into agent_ale.Agent OK)")

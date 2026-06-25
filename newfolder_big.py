"""newfolder_big.py -- a BIGGER submission model (matches pong_best's capacity) + Agent contract.

Same 2-channel [position, motion] 80x80 input and same Agent interface as newfolder.py, but a
deeper/wider CNN (3 conv layers 32/64/64 -> fc 256, ~1.7M params, on FULL 80x80 resolution) so it
has the capacity to actually represent pong_best's policy (our small 830K net topped out at net -5
/ a 95% clone). Used to DISTILL pong_best (clone the thing that beats us).
"""
import os
import numpy as np
import torch
import torch.nn as nn

UP, DOWN = 2, 3
D = 80 * 80


class Net(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 8, stride=4, padding=2), nn.ReLU(),   # (2,80,80) -> (32,20,20)
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # -> (64,10,10)
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),  # -> (64,10,10) = 6400
        )
        self.fc = nn.Linear(64 * 10 * 10, hidden)
        self.policy_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: (B, 2, 80, 80)
        h = self.conv(x).flatten(1)
        h = torch.relu(self.fc(h))
        return torch.sigmoid(self.policy_head(h)).squeeze(-1), self.value_head(h).squeeze(-1)


def features(cur, prev):
    diff = cur - prev if prev is not None else np.zeros(D, np.float32)
    return np.stack([cur.reshape(80, 80), diff.reshape(80, 80)]).astype(np.float32)


class Agent:
    def __init__(self, weights_path=None):
        self.net = Net()
        if weights_path and os.path.exists(weights_path):
            ck = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            try:
                self.net.load_state_dict(state)
            except Exception as e:
                print(f"[newfolder_big] weights don't fit this CNN ({e}) -- using random init")
        self.net.eval()
        self.prev = None

    def reset(self):
        self.prev = None

    @torch.no_grad()
    def act(self, frame):
        cur = frame.astype(np.float32).ravel()
        x = features(cur, self.prev)
        self.prev = cur
        prob, _ = self.net(torch.from_numpy(x).unsqueeze(0))
        return UP if float(prob.item()) > 0.5 else DOWN

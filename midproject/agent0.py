import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Agent:
    def __init__(self, state_dim: int, action_dim: int, weight_path: str):
        self.model = QNetwork(state_dim, action_dim).to(DEVICE)
        self.model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        self.model.eval()
        # Atari 動作名稱順序（18 維）
        self.meanings = [
            "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
            "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
            "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
            "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE",
        ]

    def act(self, obs: np.ndarray, action_mask=None, role_indicator: float = 0.0) -> int:
        """
        obs: 已正規化的 128 維 RAM；此處會附加角色旗標，輸入網路為 129 維。
        role_indicator: 上方0.0 / 下方1.0
        """
        obs_with_role = np.concatenate([obs, np.array([role_indicator], dtype=np.float32)], axis=0)
        state_t = torch.tensor(obs_with_role, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
            # 若沒有 mask，直接以 Q 值 greedy 選擇
            if action_mask is None:
                return int(torch.argmax(q_values, dim=1).item())

            mask = torch.tensor(action_mask, dtype=torch.bool, device=DEVICE)
            # 若 mask 全 0，退化成 Q 值 greedy
            if not mask.any():
                return int(torch.argmax(q_values, dim=1).item())

            # 優先找到含 "FIRE" 且合法的動作
            for idx, name in enumerate(self.meanings):
                if idx < mask.numel() and mask[idx] and "FIRE" in name:
                    return int(idx)

            # 若沒有 FIRE，優先選第一個合法且非 NOOP 的動作
            legal = torch.nonzero(mask).flatten()
            for idx in legal.tolist():
                if self.meanings[idx] != "NOOP":
                    return int(idx)

            # 若只剩 NOOP，就回傳 NOOP（至少不呆站）
            if len(legal) > 0:
                return int(legal[0])

            # 最後防呆：遮掉非法後 greedy
            q_values[0][~mask] = -1e9
            return int(torch.argmax(q_values, dim=1).item())


def load():
    state_dim = 129  # 128 RAM + 1 角色旗標
    action_dim = 18
    weight_path = "checkpoints/shared_player_final.pt"
    return Agent(state_dim, action_dim, weight_path)
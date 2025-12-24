"""
獨立 Q-學習 (IQL) 版本的雙智能體 DQN 訓練腳本。
環境：PettingZoo atari tennis_v3，使用 RAM 模式 (obs_type="ram")。
特點：
- 兩個智能體各自擁有獨立的網路、重放緩衝區與目標網路。
- 硬更新 (Hard Update) 每 1000 步同步目標網路。
- 觀測值簡單歸一化至 [0, 1]。
"""

import random
import copy
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pettingzoo.atari import tennis_v3
from tqdm import tqdm

# 裝置設定：有 GPU 則使用 CUDA，並提示當前裝置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"使用裝置: {DEVICE}")


def preprocess_observation(obs: Dict) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    將環境回傳的 observation 字典拆解：
    - 取出 RAM 狀態並除以 255 進行歸一化
    - 如果有 action_mask 也一併返回
    """
    if isinstance(obs, dict) and "observation" in obs:
        ram = obs["observation"].astype(np.float32) / 255.0
        mask = obs.get("action_mask")
    else:
        ram = np.array(obs, dtype=np.float32) / 255.0
        mask = None
    return ram, mask


class QNetwork(nn.Module):
    """簡單的 MLP Q 網路：128 -> 256 -> 256 -> action_dim。"""

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


class ReplayBuffer:
    """使用 deque 實作的經驗回放。"""

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.tensor(states, dtype=torch.float32, device=DEVICE),
            torch.tensor(actions, dtype=torch.long, device=DEVICE),
            torch.tensor(rewards, dtype=torch.float32, device=DEVICE),
            torch.tensor(next_states, dtype=torch.float32, device=DEVICE),
            torch.tensor(dones, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


class SharedDQN:
    """
    共享網路 + 自我對戰的 DQN：
    - 單一 local_net / target_net / replay_buffer，兩個角色共用。
    - 對手使用歷史快照 (self-play) 進行對戰。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_final: float = 0.01,
        exploration_fraction: float = 0.1,
        total_timesteps: int = 500_000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        buffer_size: int = 100_000,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_final = epsilon_final
        self.exploration_fraction = exploration_fraction
        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step = 0
        self.global_step = 0

        self.local_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.local_net.state_dict())

        self.optimizer = optim.Adam(self.local_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.scaler = GradScaler(enabled=DEVICE.type == "cuda")

    def _q_values(self, net: nn.Module, state: np.ndarray, action_mask: Optional[np.ndarray]) -> torch.Tensor:
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        q_values = net(state_t)
        if action_mask is not None:
            mask = torch.tensor(action_mask, dtype=torch.bool, device=DEVICE)
            invalid = ~mask
            q_values[0][invalid] = -1e9
        return q_values

    def select_action(self, state: np.ndarray, action_mask: Optional[np.ndarray]) -> int:
        """
        學習者：epsilon-greedy。
        """
        if np.random.rand() < self.epsilon:
            if action_mask is not None:
                valid_actions = np.nonzero(action_mask)[0]
                action = int(np.random.choice(valid_actions))
            else:
                action = random.randrange(self.action_dim)
        else:
            q_values = self._q_values(self.local_net, state, action_mask)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def select_action_opponent(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray],
        opponent_net: nn.Module,
        epsilon_opponent: float = 0.05,
    ) -> int:
        """
        對手：使用歷史快照網路，帶小 epsilon 讓策略共進。
        """
        if np.random.rand() < epsilon_opponent:
            if action_mask is not None:
                valid_actions = np.nonzero(action_mask)[0]
                if len(valid_actions) > 0:
                    return int(np.random.choice(valid_actions))
            return random.randrange(self.action_dim)
        q_values = self._q_values(opponent_net, state, action_mask)
        return int(torch.argmax(q_values, dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def hard_update_target(self):
        """
        硬更新 (Hard Update)：直接複製 local_net 參數到 target_net。
        """
        self.target_net.load_state_dict(self.local_net.state_dict())

    def snapshot(self) -> nn.Module:
        """
        複製當前 local_net 權重，作為歷史對手。
        """
        clone = QNetwork(self.local_net.net[0].in_features, self.action_dim).to(DEVICE)
        clone.load_state_dict(copy.deepcopy(self.local_net.state_dict()))
        clone.eval()
        return clone

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Q(s, a)
        with autocast(enabled=DEVICE.type == "cuda"):
            q_values = self.local_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_next_q = self.target_net(next_states).max(dim=1)[0]
                q_targets = rewards + (1 - dones) * self.gamma * max_next_q
            loss = nn.MSELoss()(q_values, q_targets)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.learn_step += 1
        # 每 target_update_freq 步執行一次硬更新
        if self.learn_step % self.target_update_freq == 0:
            self.hard_update_target()

        return loss.item()

    def decay_epsilon(self):
        # 線性衰減，到達 exploration_fraction * total_timesteps 時落到 epsilon_final
        decay_steps = max(1, int(self.exploration_fraction * self.total_timesteps))
        progress = min(1.0, self.global_step / decay_steps)
        self.epsilon = max(self.epsilon_final, 1.0 - (1.0 - self.epsilon_final) * progress)


def train_iql(
    num_envs: int = 4,                 # 並行環境數量
    max_steps: int = 2000,             # 單環境單局最長步數
    lr: float = 1e-4,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_final: float = 0.01,
    exploration_fraction: float = 0.1,
    total_timesteps: int = 1000_000,
    batch_size: int = 32,
    target_update_freq: int = 1000,
    buffer_size: int = 100_000,
    learning_starts: int = 100_000,
    train_freq: int = 4,
    gradient_steps: int = 1,
    save_dir: str = "checkpoints",
    snapshot_every: int = 100,
    max_snapshots: int = 5,
    early_stop_patience: int = 50,
    early_stop_target: float = 15.0,
    stagnant_limit: int = 500,
    opponent_epsilon: float = 0.05,
    opponent_refresh_every: int = 200,
):
    """
    訓練主程式：建立 parallel_env，使用共享網路 + 歷史對戰 (self-play)。
    - 單一 SharedDQN，同時扮演雙方。
    - 對手來自歷史權重快照池，提升穩定性。
    """
    envs = [tennis_v3.parallel_env(obs_type="ram") for _ in range(num_envs)]
    for idx, e in enumerate(envs):
        e.reset(seed=42 + idx)

    agent_names = envs[0].possible_agents
    action_dim = envs[0].action_space(agent_names[0]).n
    state_dim = 128  # RAM 狀態維度

    # 單一共享 DQN，自我對戰
    shared_agent = SharedDQN(
        state_dim,
        action_dim,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_final=epsilon_final,
        exploration_fraction=exploration_fraction,
        total_timesteps=total_timesteps,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        buffer_size=buffer_size,
    )
    opponent_pool = [shared_agent.snapshot()]  # 初始對手快照

    rewards_history_0 = []
    rewards_history_1 = []

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    progress = tqdm(total=total_timesteps, desc="訓練進度", ascii=True)

    total_steps = 0
    # 初始化每個環境的狀態與回合累積獎勵
    env_states = []
    env_masks = []
    env_ep_rewards = []
    env_steps = []
    for e in envs:
        obs, _ = e.reset()
        s = {}
        m = {}
        for name in agent_names:
            s[name], m[name] = preprocess_observation(obs[name])
        env_states.append(s)
        env_masks.append(m)
        env_ep_rewards.append({name: 0.0 for name in agent_names})
        env_steps.append(0)

    while total_steps < total_timesteps:
        for env_idx, env in enumerate(envs):
            if total_steps >= total_timesteps:
                break

            states = env_states[env_idx]
            masks = env_masks[env_idx]
            ep_rewards = env_ep_rewards[env_idx]

            actions = {}
            actions[agent_names[0]] = shared_agent.select_action(
                states[agent_names[0]], masks[agent_names[0]]
            )
            opponent_net = random.choice(opponent_pool)
            actions[agent_names[1]] = shared_agent.select_action_opponent(
                states[agent_names[1]],
                masks[agent_names[1]],
                opponent_net,
                epsilon_opponent=opponent_epsilon,
            )

            next_raw_obs, rewards, terminations, truncations, infos = env.step(actions)

            next_states = {}
            next_masks = {}
            dones = {}
            for name in agent_names:
                next_states[name], next_masks[name] = preprocess_observation(next_raw_obs[name])
                dones[name] = terminations[name] or truncations[name]

            for name in agent_names:
                reward = rewards[name]
                done = dones[name]
                next_state = next_states[name] if not done else np.zeros_like(states[name])
                shared_agent.store(states[name], actions[name], reward, next_state, float(done))

            if (
                len(shared_agent.replay_buffer) >= learning_starts
                and (shared_agent.global_step % train_freq == 0)
            ):
                for _ in range(gradient_steps):
                    shared_agent.learn()

            ep_rewards[agent_names[0]] += rewards[agent_names[0]]
            ep_rewards[agent_names[1]] += rewards[agent_names[1]]

            env_states[env_idx] = next_states
            env_masks[env_idx] = next_masks

            shared_agent.global_step += 1
            total_steps += 1
            progress.update(1)
            shared_agent.decay_epsilon()
            env_steps[env_idx] += 1

            # 若單局獎勵長期低迷則提前結束該局
            if (
                env_steps[env_idx] > stagnant_limit
                and abs(ep_rewards[agent_names[0]]) < 0.1
                and abs(ep_rewards[agent_names[1]]) < 0.1
            ) or all(dones.values()):
                obs, _ = env.reset()
                s = {}
                m = {}
                for name in agent_names:
                    s[name], m[name] = preprocess_observation(obs[name])
                env_states[env_idx] = s
                env_masks[env_idx] = m
                env_ep_rewards[env_idx] = {name: 0.0 for name in agent_names}
                env_steps[env_idx] = 0
            else:
                env_ep_rewards[env_idx] = ep_rewards

        # 紀錄最近一次完成的環境獎勵（近似統計）
        rewards_history_0.append(np.mean([r[agent_names[0]] for r in env_ep_rewards]))
        rewards_history_1.append(np.mean([r[agent_names[1]] for r in env_ep_rewards]))

        progress.set_postfix(
            eps=f"{shared_agent.epsilon:.3f}",
            r0=f"{rewards_history_0[-1]:.2f}",
            r1=f"{rewards_history_1[-1]:.2f}",
        )

        # 週期性加入最新快照，並適度刷新對手池避免過舊
        if shared_agent.global_step % (snapshot_every * num_envs) == 0:
            opponent_pool.append(shared_agent.snapshot())
            if len(opponent_pool) > max_snapshots:
                opponent_pool.pop(0)
        if shared_agent.global_step % (opponent_refresh_every * num_envs) == 0:
            opponent_pool = [shared_agent.snapshot()]

        if len(rewards_history_0) >= early_stop_patience:
            avg0 = np.mean(rewards_history_0[-early_stop_patience:])
            avg1 = np.mean(rewards_history_1[-early_stop_patience:])
            if avg0 >= early_stop_target and avg1 >= early_stop_target:
                print(
                    f"達標早停：最近 {early_stop_patience} 回合 avg "
                    f"{agent_names[0]}={avg0:.2f}, {agent_names[1]}={avg1:.2f}"
                )
                break

    progress.close()
    env.close()
    # 儲存最終權重
    torch.save(shared_agent.local_net.state_dict(), save_path / "shared_player_final.pt")
    print(f"訓練完成，權重已儲存至 {save_path.resolve()}")


if __name__ == "__main__":
    train_iql()


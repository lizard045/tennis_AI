import numpy as np
from pettingzoo.atari import tennis_v3
import agent0 as player0
import agent1 as player1


def preprocess(observation):
    """將 RAM 觀測正規化到 [0,1]，回傳向量與 action_mask。"""
    if isinstance(observation, dict) and "observation" in observation:
        ram = observation["observation"].astype(np.float32) / 255.0
        mask = observation.get("action_mask")
    else:
        ram = np.array(observation, dtype=np.float32) / 255.0
        mask = None
    return ram, mask


def determine_winner(done):
    p0 = done.get("player_0", 0)
    p1 = done.get("player_1", 0)
    if p0 == -1 and p1 == 1:
        return "player_1 獲勝"
    if p0 == 1 and p1 == -1:
        return "player_0 獲勝"
    if p0 == 0 and p1 == 0:
        return "平局"
    if p0 == -1 and p1 == 0:
        return "player_0 違反規則"
    if p0 == 0 and p1 == -1:
        return "player_1 違反規則"
    return "未知狀況，請檢查輸入值"


def main(render_mode="human"):
    # 載入模型（共用同一份權重）
    agent_0 = player0.load()
    agent_1 = player1.load()

    # 建立 RAM 模式環境，與訓練一致
    env = tennis_v3.env(render_mode=render_mode, obs_type="ram")
    env.reset()

    done = {"player_0": False, "player_1": False}
    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        current_agent = agent_0 if agent_name == "player_0" else agent_1

        if termination or truncation:
            action = None
            done[agent_name] = reward
        else:
            obs_vec, mask = preprocess(observation)
            action = current_agent.act(obs_vec, mask)
        env.step(action)

    env.close()
    print(determine_winner(done))


if __name__ == "__main__":
    # 如需可視化，render_mode 可設 "human"
    main(render_mode="human")
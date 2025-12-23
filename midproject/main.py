import torch
import numpy as np
import torch.nn as nn
from pettingzoo.atari import tennis_v3
import agent0 as player0
import agent1 as player1

###
# 執行請由agentplay.py執行
# 將各模型導入agent0.py與agent1.py
###
# 勝敗判斷
def determine_winner(done):
    player_0 = done["player_0"]
    player_1 = done["player_1"]
    
    # 判斷輸贏或違規
    if player_0 == -1 and player_1 == 1:
        return "player_1 獲勝"
    elif player_0 == 1 and player_1 == -1:
        return "player_0 獲勝"
    elif player_0 == 0 and player_1 == 0:
        return "平局"
    elif player_0 == -1 and player_1 == 0:
        return "player_0 違反規則"
    elif player_0 == 0 and player_1 == -1:
        return "player_1 違反規則"
    else:
        return "未知狀況，請檢查輸入值"

# 模型1與模型2載入
agent_0 = player0.load()
agent_1 = player1.load()

# 開始進行一局比賽
# 載入環境
env = tennis_v3.env(render_mode="human", obs_type="rgb_image")
env.reset()

done = {"player_0": False, "player_1": False}
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    current_agent = agent_0 if agent == "player_0" else agent_1

    if termination or truncation:
        action = None
        if agent == "player_0":
            done["player_0"] = reward
        else:
            done["player_1"] = reward
    else:
        mask = observation["action_mask"]
        observation_flat = observation["observation"].flatten()
        
        action = current_agent.act(observation_flat)

        # 確保動作是合法的
        if mask[action] == 0:
            action = np.random.choice(np.where(mask == 1)[0])

    env.step(action)

env.close()

print(determine_winner(done))
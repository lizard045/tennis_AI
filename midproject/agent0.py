import torch
import numpy as np
import torch.nn as nn
#請將模型1導入至本檔案


#代理主區塊
class Agent:
    def __init__(self, state_dim, action_dim):
        pass
    #動作決擇
    def act(self, state):
        action = None
        return action

def load():
    state_dim = 10800    #根據使用CNN或MLP自行設定 
    action_dim = 18  #動作空間

    # 初始化代理
    agent_0 = Agent(state_dim, action_dim)

    # 載入之前儲存的模型權重
    agent_0.model.load_state_dict(torch.load("dqn_step_20000.pt"))

    #導出模型1
    return agent_0
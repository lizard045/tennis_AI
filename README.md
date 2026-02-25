# Tennis IQL 自我對戰訓練專案

> **環境**：PettingZoo `tennis_v3`｜**觀測**：RGB 影像（84×84 灰階 × 4 幀堆疊）｜**演算法**：獨立 Q-學習（IQL）+ 共享網路自我對戰

## 對戰畫面預覽

![Tennis 對戰畫面](tennis.png)

上方球員（`first_0`）與下方球員（`second_0`）分別由不同模型權重控制，透過 `main.py` 進行即時對打。

---

## 專案架構

```
midproject/
├── iql_train.py          # 訓練主程式（IQL + 共享 DQN + 自我對戰）
├── agent0.py             # 推論代理人 0（ConvQNetwork，載入 shared_player_final.pt）
├── agent1.py             # 推論代理人 1（ResidualPolicy，載入 best_model.pt）
├── main.py               # 對打入口（人類渲染模式，逐步推論）
├── checkpoints/
│   ├── shared_player_final.pt   # 訓練完成的主要權重
│   ├── best_model.pt            # ResidualPolicy 訓練權重
│   └── shared_player_rgb.pt    # 備用快照
└── readme.md
```

---

## 環境需求

| 套件 | 說明 |
|------|------|
| Python 3.9+ | 基礎環境 |
| PyTorch（建議 CUDA） | 深度學習框架，有 GPU 時自動啟用 AMP |
| `pettingzoo[atari]` | 多智能體 Atari 環境 |
| `ale-py` + `autorom[accept-rom-license]` | Atari ROM 驅動 |
| `opencv-python-headless` | 影像前處理（灰階、縮放） |
| `tqdm`、`numpy` | 進度條與數值計算 |

```bash
pip install torch pettingzoo[atari] ale-py "autorom[accept-rom-license]" opencv-python-headless tqdm numpy
```

---

## 快速開始

```bash
# 訓練（RGB 影像模式，預設 800 萬步）
python iql_train.py

# 對打推論（載入雙方代理人，人類視窗渲染）
python main.py
```

---

## 完整訓練流程

### 1. 建立平行環境

```
num_envs = 2 個 tennis_v3.parallel_env(obs_type="rgb_image")
每個環境 seed=42+idx，確保可重現性
```

### 2. 影像前處理與幀堆疊

```
RGB (H,W,3) → 灰階 → resize 84×84 → uint8
每個 agent 維護長度 4 的 deque，reset 後預填零幀
stack_frames() 回傳 (4,84,84) uint8
```

### 3. 共享 DQN 架構（ConvQNetwork）

```
Input: (B, 4, 84, 84) float32（取樣時 /255 正規化）
Conv2d(4→32, 8×8, stride=4) → ReLU
Conv2d(32→64, 4×4, stride=2) → ReLU
Conv2d(64→64, 3×3, stride=1) → ReLU → Flatten
Linear(64×7×7=3136 → 512) → ReLU
Linear(512 → action_dim=18)
```

### 4. 行動選擇策略

| 角色 | 方法 | 探索方式 |
|------|------|----------|
| 學習者（上方 `first_0`） | `SharedDQN.select_action` | epsilon-greedy（線性衰減 1.0→0.01） |
| 對手（下方 `second_0`） | `select_action_opponent` | 50% 最強快照 + 50% 歷史隨機，固定 ε=0.10 |

### 5. 環境步進與經驗回放

```
env.step(actions) → next_obs, rewards, terminations, truncations
探索初期（前 200,000 步）：reward × 3.0（explore_bonus_scale）
寫入 ReplayBuffer 以 uint8 儲存，取樣時 /255 轉 float32
```

### 6. 觸發學習條件

```
len(buffer) >= learning_starts(20,000)
且 global_step % train_freq(4) == 0
執行 gradient_steps(1) 次反向傳播
每 target_update_freq(1000) 步硬更新目標網路
```

### 7. 對手池維護

```
每 snapshot_every(100) × num_envs 步：加入最新快照
池超過 max_snapshots(5)：FIFO 移除最舊
每 opponent_refresh_every(200) × num_envs 步：用當前網路刷新 best_opponent
```

### 8. 回合管理與早停

```
episode 步數 > stagnant_limit(2000) 且雙方獎勵 ≈ 0 → 提前 reset（防卡球）
最近 early_stop_patience(50) 回合雙方平均獎勵 >= early_stop_target(15.0) → 停止訓練
```

### 9. 儲存模型

```
正常完成 → checkpoints/shared_player_final.pt
Ctrl+C 中斷 → checkpoints/shared_player_step{global_step}.pt
```

---

## 核心數學原理

### Q 函數與 Bellman 目標

$$y = r + \gamma \max_{a'} Q_{\text{target}}(s', a';\, \theta^-)$$

$$L(\theta) = \frac{1}{N}\sum_{i=1}^{N}\left(Q(s_i,a_i;\theta) - y_i\right)^2$$

### 目標網路硬更新

$$\theta^- \leftarrow \theta \quad \text{（每 1000 步）}$$

穩定訓練目標，降低 Q 值震盪。

### Epsilon-Greedy 線性衰減

$$\epsilon_t = \max\!\Big(\epsilon_{\min},\ 1 - (1-\epsilon_{\min}) \cdot \frac{t}{T_{\text{decay}}}\Big)$$

其中 $T_{\text{decay}} = \text{exploration\_fraction} \times \text{total\_timesteps}$。

### 對手池混合策略

- **50%** 使用 `best_opponent`（定期以最新網路刷新）
- **50%** 隨機從歷史快照池抽取

目標是讓學習者面對多樣且持續升級的對手，避免策略過擬合。

---

## 推論架構（對打模式）

`main.py` 載入兩個不同架構的代理人進行對打：

| 角色 | 檔案 | 網路架構 | 權重 |
|------|------|----------|------|
| `first_0`（上方） | `agent1.py` | ResidualPolicy（殘差卷積 + Actor/Critic 雙分支） | `checkpoints/best_model.pt` |
| `second_0`（下方） | `agent0.py` | ConvQNetwork（Nature DQN 架構） | `checkpoints/shared_player_final.pt` |

**ResidualPolicy 架構（agent1.py）：**
```
Input: (1, 4, 84, 84)
Conv(4→32, 3×3, stride=2) + 2×ResidualBlock(32)
Conv(32→64, 3×3, stride=2) + 2×ResidualBlock(64)
Conv(64→64, 3×3, stride=2) + 2×ResidualBlock(64)
Flatten → Linear(7744→512) → ReLU
Actor: Linear(512→18)   ← 動作輸出
Critic: Linear(512→1)   ← 值函數（推論時不使用）
```

**行動選擇優先順序（兩個代理人共用）：**
1. 動作中含 `FIRE` 的合法動作（優先擊球）
2. 合法且非 `NOOP` 的動作
3. 任意合法動作
4. 最後防呆：Greedy argmax（遮掉非法動作後）

---

## 重要超參數（預設值）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `total_timesteps` | 8,000,000 | 總訓練步數 |
| `num_envs` | 2 | 並行環境數 |
| `FRAME_STACK` | 4 | 堆疊幀數 |
| `buffer_size` | 20,000 | 回放緩衝大小 |
| `learning_starts` | 20,000 | 開始學習的緩衝下限 |
| `batch_size` | 16 | 批次大小 |
| `train_freq` | 4 | 每 N 步學習一次 |
| `gradient_steps` | 1 | 每次學習反傳次數 |
| `lr` | 1e-4 | Adam 學習率 |
| `gamma` | 0.99 | 折扣因子 |
| `epsilon_final` | 0.01 | epsilon 衰減下限 |
| `exploration_fraction` | 0.1 | epsilon 衰減佔總步數比例 |
| `target_update_freq` | 1000 | 硬更新目標網路頻率 |
| `opponent_epsilon` | 0.10 | 對手固定探索率 |
| `snapshot_every` | 100 | 快照間隔（回合數） |
| `max_snapshots` | 5 | 對手池容量上限 |
| `opponent_refresh_every` | 200 | 刷新最強對手間隔 |
| `stagnant_limit` | 2000 | 卡球偵測步數上限 |
| `explore_bonus_scale` | 3.0 | 早期探索獎勵倍率 |
| `explore_bonus_steps` | 200,000 | 探索獎勵持續步數 |
| `early_stop_patience` | 50 | 早停觀察回合數 |
| `early_stop_target` | 15.0 | 早停目標平均獎勵 |

---

## 常見調參建議

- **快速驗證**：降低 `total_timesteps`（如 500,000）或 `num_envs=1`。
- **回合過短/卡球**：提高 `max_steps` 或 `stagnant_limit`，放大 `opponent_epsilon`。
- **記憶體不足**：縮小 `buffer_size`（如 10,000）或 `batch_size`（如 8）。
- **對手太弱**：增加 `max_snapshots` 或縮短 `opponent_refresh_every`。
- **想看訓練曲線**：接入 TensorBoard，在 `learn()` 後記錄 `loss`、`epsilon`、`reward`。

---

## Docker 快速建環境

```bash
docker run --gpus all --shm-size=1g \
  -e DISPLAY=10.1.2.15:0 \
  -v "C:\Users\snowy\OneDrive\桌面\Lizard\project\intelligent_AI\midproject:/workspace" \
  -it pettingzoo-atari bash
```

---

## 注意事項

- 本版採 `obs_type="rgb_image"`，勿混用 RAM 版（`obs_type="ram"`）的超參或狀態維度。
- Replay Buffer 以 `uint8` 存放影像，取樣時必須除以 `255.0`（程式已處理）。
- `main.py` 中 `agent0.py` 扮演下方 `second_0`，`agent1.py` 扮演上方 `first_0`，角色與檔名相反，請注意對應關係。
- 遠端渲染或使用 Docker 時，請確認 `DISPLAY` 環境變數與 X server 正常運作。

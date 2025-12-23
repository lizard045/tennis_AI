# 共享網路 + 自我對戰 (Self-Play) DQN 訓練指引

本專案在 PettingZoo `tennis_v3`（RAM 模式）上，改為**單一共享 DQN 網路**並透過歷史對戰（self-play）訓練雙方。重放緩衝與網路皆共用，對手來自歷史快照池，提升對抗穩定性。

## 環境需求
- Python 3.9+
- `pettingzoo[atari]`
- `torch`
- `numpy`

安裝範例（依需求調整環境）：
```bash
pip install "pettingzoo[atari]" torch numpy
pip install tqdm
```

## 檔案說明
- `iql_train.py`：完整訓練腳本。
- `agent0.py` / `agent1.py` / `main.py`：原有範例程式，與本訓練腳本相互獨立。

## 訓練流程摘要
1. 建立 `tennis_v3.parallel_env(obs_type="ram")`，取得 128 維 RAM 狀態並除以 255 正規化。
2. 使用單一 `SharedDQN`（local/target/replay 共用），同時為兩邊選動作；學習端採 epsilon-greedy，對手端以歷史快照 greedy。
3. 雙方經驗統一寫入共享 Replay Buffer；每 1000 步執行一次「硬更新 (Hard Update)」同步 target_net。
4. 每回合衰減 epsilon（0.995，下限 0.01），預設 1000 回合；`tqdm` 顯示進度，每 100 回合輸出最近 100 回合平均獎勵。
5. 週期性將當前網路快照放入對手池（預設每 100 回合，最多 5 個），自我對戰時隨機挑選一個歷史對手。

## 執行方式
```bash
python iql_train.py
```

## 產出與調整
- 訓練結束後會將共享網路權重儲存到 `checkpoints/shared_player_final.pt`。
- 可調整學習率、回合數、硬更新頻率、`snapshot_every`、`max_snapshots`，或修改 `save_dir` 變更輸出路徑。


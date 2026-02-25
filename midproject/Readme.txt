test_env.py      環境測試程式:用於測試安裝環境是否正確
Dockerfile       環境安裝指令檔，如要安裝其他套件使用請 滑鼠右鍵>記事本開啟>編輯完畢>儲存
                 (切記Dockerfile檔不能有副檔名，不要儲存成txt檔)
執行指令
docker run --gpus all --shm-size=1g -e DISPLAY=10.1.2.15:0  -v "C:\Users\snowy\OneDrive\桌面\Lizard\project\intelligent_AI\midproject:/workspace" -it pettingzoo-atari bash


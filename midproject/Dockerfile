# 基底改為含 CUDA Runtime 的映像，方便使用 GPU
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 避免互動安裝提示
ENV DEBIAN_FRONTEND=noninteractive

# 安裝 Python 與系統依賴（編譯工具、圖形套件、swig、cmake）
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    cmake g++ swig zlib1g-dev git wget unzip libgl1 libgl1-mesa-dri xvfb && \
    rm -rf /var/lib/apt/lists/*

# 讓 python / pip 指向系統 python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 設定工作目錄
WORKDIR /workspace

# 升級 pip、setuptools、wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 安裝 ALE、PettingZoo Atari、AutoROM、Gymnasium、Matplotlib、tqdm
RUN pip install --no-cache-dir "ale-py==0.8.1"
RUN pip install --no-cache-dir "pettingzoo[atari]" autorom[accept-rom-license] gymnasium matplotlib tqdm

# 安裝 OpenCV (headless 模式)
RUN pip install --no-cache-dir opencv-python-headless

# 安裝 PyTorch GPU 版（含 torchvision、torchaudio），使用官方 CUDA 12.1 來源
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir pyvirtualdisplay

# 自動同意 ROM 授權
RUN AutoROM --accept-license

# 設定 headless render
ENV DISPLAY=:99

# 預設進入 bash
CMD ["bash"]

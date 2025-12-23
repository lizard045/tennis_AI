# 使用 Python 3.9 slim 作為基底
FROM python:3.9-slim

# 安裝系統依賴（編譯工具、圖形套件、swig、cmake）
RUN apt-get update && apt-get install -y \
    cmake g++ swig zlib1g-dev git wget unzip libgl1 libgl1-mesa-dri xvfb && \
    rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /workspace

# 升級 pip、setuptools、wheel、build
RUN pip install --upgrade pip setuptools wheel build

# 安裝 ALE、PettingZoo Atari、AutoROM、Gymnasium、Matplotlib
RUN pip install "ale-py==0.8.1"
RUN pip install "pettingzoo[atari]" autorom[accept-rom-license] gymnasium matplotlib

# 安裝 OpenCV (headless 模式)
RUN pip install opencv-python-headless

# 安裝 PyTorch CPU 版本（含 torchvision、torchaudio）
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install pyvirtualdisplay

# 自動同意 ROM 授權
RUN AutoROM --accept-license

# 設定 headless render
ENV DISPLAY=:99

# 預設進入 bash
CMD ["bash"]

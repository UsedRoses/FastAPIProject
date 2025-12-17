# 基础镜像
FROM python:3.10-slim

# 环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# 1. 安装系统依赖 (保持不变)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    cmake \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. 升级 pip 并预装构建工具 (保持不变)
RUN pip install --upgrade pip && \
    pip install setuptools wheel ninja

# ==================== 核心修改开始 ====================

# 3. 【重点】优先单独安装 PyTorch 和 Numpy
# YOLOX 和 cython-bbox 极其依赖这两个库，必须先装好，把底座打牢
RUN pip install --no-cache-dir \
    numpy>=1.23.5 \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu118

# 4. 【重点】优先单独安装 cython
# cython-bbox 需要它
RUN pip install --no-cache-dir cython

# 5. 复制 requirements.txt
COPY requirements.txt .

# 6. 安装剩余依赖
# 注意：这里会安装 YOLOX，此时环境里已经有 torch 了，所以不会报错
RUN pip install --no-cache-dir -r requirements.txt

# ==================== 核心修改结束 ====================

# 7. 复制脚本和模型
COPY scripts/ /app/scripts/
COPY app/models/ /app/app/models/
RUN python /app/scripts/download_weights.py

# 8. 复制代码
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
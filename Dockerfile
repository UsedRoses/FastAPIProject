FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# 1. 安装系统依赖
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

# 2. 预装构建工具
RUN pip install --upgrade pip && \
    pip install setuptools wheel ninja

# 3. 单独安装 Torch (利用缓存)
RUN pip install --no-cache-dir \
    numpy>=1.23.5 \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu118

# 4. 单独安装 Cython
RUN pip install --no-cache-dir cython

# 5. 复制并安装 requirements.txt (不含 YOLOX)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==================== 关键修改 ====================
# 6. 单独安装 YOLOX，并禁用构建隔离
# --no-build-isolation: 告诉 pip 不要创建临时环境，直接使用系统里已经装好的 torch
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/Megvii-BaseDetection/YOLOX.git
# ================================================

# 7. 下载脚本和模型
COPY scripts/ /app/scripts/
COPY app/models/ /app/app/models/
RUN python /app/scripts/download_weights.py

# 8. 复制代码
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
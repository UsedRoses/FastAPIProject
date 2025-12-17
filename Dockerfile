# 基础镜像：Python 3.10 Slim (轻量且稳定)
FROM python:3.10-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# 防止 pip 提示升级
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# 1. 安装系统级依赖 (这一步最重要)
# build-essential: 包含 gcc/g++，编译 cython-bbox 必须
# ffmpeg: 视频处理必须
# libgl1/libsm6: OpenCV 必须
# git: 拉取 YOLOX 源码必须
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

# 2. 升级 pip 并预先安装编译所需的 Python 包
# 解释：cython-bbox 安装时需要用到 cython 和 numpy，必须先装好它们
RUN pip install --upgrade pip && \
    pip install cython numpy setuptools wheel

# 3. 复制依赖文件
COPY requirements.txt .

# 4. 安装其余依赖
# 使用 --no-cache-dir 减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制下载脚本和模型目录 (确保模型能被打包进去)
# 假设你的目录结构里有 scripts/download_weights.py
COPY scripts/ /app/scripts/
COPY app/models/ /app/app/models/

# 6. 运行下载脚本 (构建时下载模型)
# 确保你的 download_weights.py 里下载的是 yolox_x.pth
RUN python /app/scripts/download_weights.py

# 7. 复制剩余代码
COPY . .

# 8. 暴露端口
EXPOSE 8000

# 9. 启动服务
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
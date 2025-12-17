import os
import urllib.request
import argparse

# 官方权重映射表 (Apache 2.0)
MODELS = {
    "yolox_l": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth",
    "yolox_x": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth"
}


def download_model(model_name="yolox_x"):
    if model_name not in MODELS:
        print(f"错误: 未知的模型名称 {model_name}")
        return

    url = MODELS[model_name]
    filename = f"{model_name}.pth"

    # 保存路径: app/models/yolox_l.pth
    save_dir = os.path.join(os.path.dirname(__file__), "../app/models")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        print(f"检测到模型已存在: {save_path}，跳过下载。")
        return

    print(f"--- 开始下载高精度模型: {model_name} ---")
    print(f"下载源: {url}")

    try:
        # 添加 User-Agent 避免某些网络环境被 GitHub 拒绝
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(url, save_path)
        print(f"下载成功！")
        print(f"保存在: {save_path}")
    except Exception as e:
        print(f"下载失败: {e}")
        exit(1)


if __name__ == "__main__":
    # 默认下载 yolox_l，追求极致可改为 yolox_x
    download_model("yolox_x")
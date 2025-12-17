# 封装 YOLOX (Detection)

import torch
import numpy as np
import cv2
import os
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess


class YOLOXDetector:
    def __init__(self, model_path, model_name="yolox-l", device="cpu"):
        # 自动判断设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using Device: {self.device}")  # 打印出来确认一下
        # 1. 获取模型实验配置
        # YOLOX 的设计模式需要先加载"实验配置(Experiment)"，再加载权重
        # 这里为了简化，我们根据 model_name 手动设置 depth 和 width
        # yolox-l: depth=1.0, width=1.0
        # yolox-x: depth=1.33, width=1.25
        if "yolox_x" in model_path:
            self.exp = get_exp(None, "yolox_x")
        else:
            self.exp = get_exp(None, "yolox_l")  # 默认 L

        self.model = self.exp.get_model()

        # 2. 加载权重
        print(f"Loading weights form {model_path}")
        ckpt = torch.load(model_path, map_location=device)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device)
        self.model.eval()

        # 3. 预处理参数
        self.test_size = (640, 640)  # YOLOX 标准输入尺寸
        self.preproc = ValTransform(legacy=False)

    def detect(self, img):
        """
        输入: 原始 OpenCV 图片 (H, W, 3)
        输出: 检测框列表 [[x1, y1, x2, y2, score, class_id], ...]
        """
        height, width = img.shape[:2]

        # 预处理：Resize 和 Pad
        img_info = {"id": 0}
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img_info["height"], self.test_size[1] / img_info["width"])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float().to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            # 后处理：NMS (非极大值抑制)
            outputs = postprocess(
                outputs,
                num_classes=80,
                conf_thre=0.25,
                nms_thre=0.45,
                class_agnostic=True
            )

        detections = []
        if outputs[0] is not None:
            output = outputs[0].cpu().numpy()
            # 将坐标还原回原图尺寸
            output[:, 0:4] /= ratio

            # 过滤：只保留"人" (COCO class_id = 0)
            # output 格式: [x1, y1, x2, y2, obj_conf, class_conf, class_pred]
            for det in output:
                # det[6] 是类别索引
                if int(det[6]) == 0:
                    # 重新组合: x1, y1, x2, y2, score
                    score = det[4] * det[5]
                    detections.append([det[0], det[1], det[2], det[3], score])

        # 强制转换为 numpy 数组
        final_dets = np.array(detections)

        # 如果数组为空，或者变成了一维数组（防止奇怪的边缘情况），强制 reshape 成 (N, 5)
        # 如果是空的，变成 (0, 5)；如果不为空，变成 (N, 5)
        if len(final_dets) == 0:
            return np.empty((0, 5))

        return final_dets
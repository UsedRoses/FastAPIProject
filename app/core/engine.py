# 剪辑主逻辑 (Video Processing Pipeline)

import cv2
import numpy as np
import subprocess
import os
import sys
from loguru import logger

from app.core.detector import YOLOXDetector
from app.core.tracker import VideoTracker
from app.utils.smoothing import OneEuroFilter

# --- 新增：运镜模式预设 ---
# min_cutoff: 越小画面越稳（死区越大），但延迟越高
# beta: 越大跟随越快（灵敏度高），但容易抖动
SMOOTHING_PRESETS = {
    "fast":   {"min_cutoff": 1.0,   "beta": 0.05},   # 【运动模式】户外、跑步：响应极快，允许少量抖动
    "normal": {"min_cutoff": 0.05,  "beta": 0.005},  # 【标准模式】vlog、日常：平衡点
    "stable": {"min_cutoff": 0.005, "beta": 0.0005}  # 【访谈模式】讲座、新闻：如定海神针，几乎不动，除非大幅移动
}

class SmartReframer:
    def __init__(self, model_path, device="cuda"):
        # 初始化 AI 模型
        self.detector = YOLOXDetector(model_path, model_name="yolox-x", device=device)
        self.tracker = VideoTracker()

        # 平滑器将在 process_video 中根据模式初始化
        # 分别定义 X 和 Y 的平滑器
        self.smoother_x = None
        self.smoother_y = None

    def _get_video_info(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return width, height, fps, total_frames

    def _select_main_subject(self, tracks, locked_id):
        """
        主角选择逻辑：
        1. 如果有锁定的 ID 且该 ID 还在画面内，继续选它。
        2. 如果锁定 ID 丢失，选择画面中面积最大的人作为新主角。
        """
        if not tracks:
            return None, locked_id

        # 优先锁定之前的主角
        if locked_id is not None:
            for t in tracks:
                if t['id'] == locked_id:
                    return t, locked_id

        # 如果没找到，或者还没锁定，找面积最大的
        # t['bbox'] = [x, y, w, h]
        best_track = max(tracks, key=lambda t: t['bbox'][2] * t['bbox'][3])
        return best_track, best_track['id']

    def process_video(self, input_path, output_path, ratio_str="9:16", mode="normal"):
        """
        核心流程入口
        ratio_str: "9:16", "4:3", "1:1", "16:9" 等字符串
        mode: "fast", "normal", "stable"
        """
        logger.info(f"开始处理视频: {input_path} | 比例: {ratio_str} | 模式: {mode}")

        # 1. 获取视频信息
        src_w, src_h, fps, total_frames = self._get_video_info(input_path)

        # --- 1. 智能尺寸计算 ---
        try:
            r_w, r_h = map(float, ratio_str.split(":"))
        except:
            r_w, r_h = 9.0, 16.0

        src_ratio = src_w / src_h
        target_ratio = r_w / r_h

        if target_ratio < src_ratio:
            # 情况 A: 目标比原片窄 (例如 16:9 -> 9:16)
            # 策略: 高度撑满，裁宽度 (左右运镜)
            target_h = src_h
            target_w = int(src_h * target_ratio)
        else:
            # 情况 B: 目标比原片宽 (例如 9:16 -> 16:9)
            # 策略: 宽度撑满，裁高度 (上下运镜)
            target_w = src_w
            target_h = int(src_w / target_ratio)

        logger.info(f"尺寸: {src_w}x{src_h} -> {target_w}x{target_h}")

        # --- 2. 初始化双轴滤波器 ---
        preset = SMOOTHING_PRESETS.get(mode, SMOOTHING_PRESETS["normal"])
        logger.info(f"使用平滑参数: {preset}")

        # X轴初始位置：画面中心
        self.smoother_x = OneEuroFilter(t0=0, x0=src_w / 2, min_cutoff=preset["min_cutoff"], beta=preset["beta"])
        # Y轴初始位置：画面中心
        self.smoother_y = OneEuroFilter(t0=0, x0=src_h / 2, min_cutoff=preset["min_cutoff"], beta=preset["beta"])
        # --- 第一阶段：分析 (Pass 1 - Analysis) ---
        logger.info("阶段 1/2: 智能分析运镜路径...")

        camera_path = []  # 存储每一帧的 crop_x 坐标
        locked_id = None
        cap = cv2.VideoCapture(input_path)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break

            # 1. 检测
            detections = self.detector.detect(frame)

            # 2. 跟踪
            tracks = self.tracker.update(detections, (src_h, src_w))

            # 3. 锁定主角
            subject, locked_id = self._select_main_subject(tracks, locked_id)

            # 4. 计算目标中心点
            if subject:
                # 目标的中心 x
                tx, ty = subject['center'] # 获取 x 和 y
            else:
                # 没人？缓慢回归到画面正中心
                tx, ty = src_w / 2, src_h / 2 # 回归中心


            # timestamp 使用帧号/FPS
            timestamp = i / fps

            # 5. 应用 One-Euro 滤波 (去抖动)
            # --- 3. 双轴滤波与计算 ---
            sx = self.smoother_x(timestamp, tx)
            sy = self.smoother_y(timestamp, ty)

            # 计算左上角坐标 (X轴)
            crop_x = sx - (target_w / 2)
            crop_x = max(0, min(crop_x, src_w - target_w))

            # 计算左上角坐标 (Y轴)
            crop_y = sy - (target_h / 2)
            crop_y = max(0, min(crop_y, src_h - target_h))

            # 存入路径 (x, y)
            camera_path.append((int(crop_x), int(crop_y)))

            if i % 100 == 0:
                logger.info(f"Analysis: {i}/{total_frames} frames processed.")

        cap.release()
        logger.info("路径分析完成。")

        # --- 第二阶段：渲染 (Pass 2 - Rendering) ---
        logger.info("阶段 2/2: FFmpeg 高质量渲染...")

        self._render_with_ffmpeg(input_path, output_path, camera_path, target_w, target_h, fps)

        logger.success(f"处理完成！输出文件: {output_path}")
        return output_path

    def _render_with_ffmpeg(self, input_path, output_path, camera_path, w, h, fps):
        """
        使用 FFmpeg Pipe 模式进行渲染。
        Python 负责 Crop，FFmpeg 负责 Encode。
        这样可以复用 FFmpeg 强大的编码器和音频处理能力。
        """
        # 构建 FFmpeg 命令
        # -c:v libx264 -preset slow -crf 18: 保证高质量
        # -map 0:v -map 1:a: 视频来自 pipe(0)，音频来自原文件(1)
        video_encoder = 'h264_nvenc'

        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-i', input_path,
            '-map', '0:v', '-map', '1:a',
            '-c:v', video_encoder,
            '-preset', 'p4',
            '-b:v', '5M',
            '-c:a', 'aac', '-b:a', '192k',
            '-shortest',
            output_path
        ]

        # 启动子进程
        # bufsize 设大一点提高性能
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=10 ** 7)

        cap = cv2.VideoCapture(input_path)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 安全检查：防止路径数据比视频短
            if frame_idx >= len(camera_path):
                break

            crop_x, crop_y = camera_path[frame_idx]

            # --- 核心裁剪 ---
            # numpy 切片: [y_start:y_end, x_start:x_end]
            crop_img = frame[crop_y : crop_y + h, crop_x : crop_x + w]

            # 写入管道
            try:
                process.stdin.write(crop_img.tobytes())
            except BrokenPipeError:
                logger.error("FFmpeg pipe broke unexpectedly.")
                break

            frame_idx += 1

        cap.release()
        process.stdin.close()
        process.wait()  # 等待 FFmpeg 编码结束
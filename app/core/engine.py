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


class SmartReframer:
    def __init__(self, model_path, device="cuda"):
        # 初始化 AI 模型
        self.detector = YOLOXDetector(model_path, model_name="yolox-x", device=device)
        self.tracker = VideoTracker()

        # 运镜平滑参数 (经过调试的电影级参数)
        # min_cutoff: 越小越平滑，0.05 类似于斯坦尼康稳定器
        # beta: 响应速度，0.005 保证快速移动时能跟上
        self.smoother = OneEuroFilter(t0=0, x0=0, min_cutoff=0.05, beta=0.005)

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

        # 尝试寻找之前锁定的 ID
        if locked_id is not None:
            for t in tracks:
                if t['id'] == locked_id:
                    return t, locked_id

        # 如果没找到，或者还没锁定，找面积最大的
        # t['bbox'] = [x, y, w, h]
        best_track = max(tracks, key=lambda t: t['bbox'][2] * t['bbox'][3])
        return best_track, best_track['id']

    def process_video(self, input_path, output_path, target_ratio=(9, 16)):
        """
        核心流程入口
        """
        logger.info(f"开始处理视频: {input_path}")

        # 1. 获取视频信息
        src_w, src_h, fps, total_frames = self._get_video_info(input_path)

        # 计算目标裁剪尺寸 (保持高度不变，计算宽度)
        # 例如: 1080p (1920x1080) -> 9:16 -> 607x1080
        target_h = src_h
        target_w = int(src_h * target_ratio[0] / target_ratio[1])

        if target_w > src_w:
            # 如果原始视频不够宽（比如本来就是竖屏），则反向逻辑或保持原样
            logger.warning("目标宽度大于原始宽度，强制适应。")
            target_w = src_w

        logger.info(f"原尺寸: {src_w}x{src_h}, 裁剪目标: {target_w}x{target_h}")

        # --- 第一阶段：分析 (Pass 1 - Analysis) ---
        logger.info("阶段 1/2: 智能分析运镜路径...")

        camera_path = []  # 存储每一帧的 crop_x 坐标
        locked_id = None

        cap = cv2.VideoCapture(input_path)

        # 预热滤波器：初始位置设为画面中心
        current_x = src_w / 2

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
                target_center_x = subject['center'][0]
            else:
                # 没人？缓慢回归到画面正中心
                target_center_x = src_w / 2

            # 5. 应用 One-Euro 滤波 (去抖动)
            # timestamp 使用帧号/FPS
            timestamp = i / fps
            smoothed_center_x = self.smoother(timestamp, target_center_x)

            # 6. 计算左上角裁剪坐标 (Clamp 边界约束)
            # 这种算法保证框永远不出界
            crop_x = smoothed_center_x - (target_w / 2)
            crop_x = max(0, min(crop_x, src_w - target_w))

            camera_path.append(int(crop_x))

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
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',  # 输入分辨率 (裁剪后的)
            '-pix_fmt', 'bgr24',  # OpenCV 默认格式
            '-r', str(fps),
            '-i', '-',  # Input 0: 来自 Python 的管道
            '-i', input_path,  # Input 1: 原视频 (用于取音频)
            '-map', '0:v',
            '-map', '1:a',  # 只要音频流
            '-c:v', 'libx264',
            '-preset', 'slow',  # 慢速编码，高质量
            '-crf', '18',  # 视觉无损级别
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',  # 以最短的流为结束（防止音频比视频长）
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

            crop_x = camera_path[frame_idx]

            # --- 核心裁剪 ---
            # numpy 切片: [y_start:y_end, x_start:x_end]
            crop_img = frame[0:h, crop_x: crop_x + w]

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
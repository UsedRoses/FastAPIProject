# 剪辑主逻辑 (Video Processing Pipeline)

import cv2
import subprocess
import os
from loguru import logger
from collections import defaultdict
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

# COCO 数据集 80 类名称映射表
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
)


# 辅助函数：把名字转 ID
def get_ids_by_names(names_str):
    if names_str.strip().lower() == "all":
        return [-1]

    # 输入 "cat,dog" -> 输出 [15, 16]
    target_ids = []
    # 默认只找人
    if not names_str:
        return [0]

    for name in names_str.split(","):
        name = name.strip().lower()
        try:
            idx = COCO_CLASSES.index(name)
            target_ids.append(idx)
        except ValueError:
            logger.warning(f"未知类别: {name}，已忽略。")

    # 如果没找到任何合法的，兜底找人
    if not target_ids:
        logger.warning("未找到有效类别，回退到默认(person)")
        return [0]
    return target_ids

class SmartReframer:
    def __init__(self, model_path, device="cuda"):
        # 初始化 AI 模型
        self.detector = YOLOXDetector(model_path, model_name="yolox-x", device=device)

        # 平滑器将在 process_video 中根据模式初始化
        # 分别定义 X 和 Y 的平滑器
        self.smoother_x = None
        self.smoother_y = None

        self.tracker = None

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

    # --- 智能重心计算 (解决站立/坐着的问题) ---
    def _calc_smart_center(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        center_x = x1 + w / 2

        # 智能垂直重心 (Head Bias):
        # 不瞄准 bbox 的几何中心 (0.5)，而是瞄准上部 35% 处 (0.35)
        # 这样无论是站着还是坐着，镜头都会偏向头部和胸部，而不是肚子
        smart_center_y = y1 + (h * 0.35)

        return center_x, smart_center_y

    def process_video(self, input_path, output_path, ratio_str="9:16", mode="normal", detect_target="person", multi_subject=False, split_screen=False):
        """
        核心流程入口
        ratio_str: "9:16", "4:3", "1:1", "16:9" 等字符串
        mode: "fast", "normal", "stable"

        根据 multi_subject 决定走单人模式还是多人模式
        """
        self.tracker = VideoTracker()

        logger.info(f"开始处理视频: {input_path} | 比例: {ratio_str} | 模式: {mode} | 目标: {detect_target} | 分镜拆条: {multi_subject}")
        if multi_subject:
            # 多主体模式, 并且新增分屏
            return self._process_multi_mode(input_path, output_path, ratio_str, mode, detect_target, split_screen)
        else:
            # 单主体模式, 为了统一接口, 单路径也包在列表里
            path = self._process_single_mode(input_path, output_path, ratio_str, mode, detect_target)
            return [path]

    # ==================== 模式 A: 原有的单人逻辑 ====================
    def _process_single_mode(self, input_path, output_path, ratio_str, mode, detect_target):
        logger.info(f"=== 单主体模式: {input_path} ===")
        # 1. 获取视频信息
        src_w, src_h, fps, total_frames = self._get_video_info(input_path)

        # 初始化滤波器
        self._init_smoothers(src_w, src_h, mode)
        # --- 解析类别 ID ---
        target_ids = get_ids_by_names(detect_target)
        logger.info(f"追踪类别 ID: {target_ids}")

        target_w, target_h = self._calc_target_size(src_w, src_h, ratio_str)

        logger.info(f"尺寸: {src_w}x{src_h} -> {target_w}x{target_h}")

        # --- 第一阶段：分析  ---
        logger.info("阶段 1/2: 智能分析运镜路径...")

        camera_path = []  # 存储每一帧的 crop_x 坐标
        locked_id = None
        cap = cv2.VideoCapture(input_path)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break

            # 1. 检测
            detections = self.detector.detect(frame, target_ids=target_ids)

            # 2. 跟踪
            tracks = self.tracker.update(detections, (src_h, src_w))

            # 3. 锁定主角
            subject = None
            if locked_id is not None:
                for t in tracks:
                    if t['id'] == locked_id: subject = t; break
            if not subject and tracks:
                subject = max(tracks, key=lambda t: t['bbox'][2] * t['bbox'][3])
                locked_id = subject['id']

            if subject:
                # 使用智能重心
                tx, ty = self._calc_smart_center(subject['bbox'])
            else:
                tx, ty = src_w / 2, src_h / 2

            # timestamp 使用帧号/FPS
            timestamp = i / fps

            # --- 双轴滤波与计算 ---
            sx = self.smoother_x(timestamp, tx)
            sy = self.smoother_y(timestamp, ty)

            # 存入路径 (x, y)
            camera_path.append(self._calc_crop_xy(sx, sy, target_w, target_h, src_w, src_h))

            if i % 100 == 0:
                logger.info(f"Analysis: {i}/{total_frames} frames processed.")

        cap.release()
        logger.info("路径分析完成。")

        # --- 第二阶段：渲染 (Pass 2 - Rendering) ---
        logger.info("阶段 2/2: FFmpeg 高质量渲染...")

        self._render_with_ffmpeg(input_path, output_path, camera_path, target_w, target_h, fps)

        logger.success(f"处理完成！输出文件: {output_path}")
        return output_path

    # ==================== 模式 B: 新增的多人拆分逻辑 ====================
    def _process_multi_mode(self, input_path, output_path, ratio_str, mode, detect_target, split_screen):
        logger.info(f"=== 多主体拆分模式: {input_path} ===")
        src_w, src_h, fps, total_frames = self._get_video_info(input_path)
        target_w, target_h = self._calc_target_size(src_w, src_h, ratio_str)
        target_ids = get_ids_by_names(detect_target)
        logger.info(f"追踪类别 ID: {target_ids}")

        # 1. 第一遍全量扫描：记录所有 ID 的原始轨迹
        # raw_tracks_history = { id: { frame_idx: (x, y) } }
        raw_tracks_history = defaultdict(dict)

        cap = cv2.VideoCapture(input_path)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break

            detections = self.detector.detect(frame, target_ids=target_ids)
            tracks = self.tracker.update(detections, (src_h, src_w))

            # 记录这一帧出现的所有 ID 及其位置
            for t in tracks:
                tid = t['id']
                raw_tracks_history[tid][i] = t['bbox']

            if i % 100 == 0: logger.info(f"Scanning All: {i}/{total_frames}")
        cap.release()

        # 2. 筛选
        absolute_min_frames = 30
        sorted_candidates = sorted(
            [(tid, len(h)) for tid, h in raw_tracks_history.items() if len(h) > absolute_min_frames],
            key=lambda x: x[1],
            reverse=True
        )
        valid_ids = [x[0] for x in sorted_candidates]
        logger.info(f"检测到的主体排序: {sorted_candidates}")

        if not valid_ids:
            logger.warning("未检测到常驻主体，回退到单人模式")
            path = self._process_single_mode(input_path, output_path, ratio_str, mode, detect_target)
            return [path]

        logger.info(f"检测到 {len(valid_ids)} 个常驻主体: {valid_ids}")

        # --- 分屏模式开启 (Split Screen) ---
        if split_screen:
            logger.info(f"生成智能分屏视频 (Top 2 Subjects)")

            # 选出 Top 2 (按出现时长排序)
            sorted_ids = sorted(valid_ids, key=lambda x: len(raw_tracks_history[x]), reverse=True)

            # 下半屏(主视角): ID 1
            id_bottom = sorted_ids[0]
            # 上半屏(副视角): ID 2 (如果没有第二个ID，就用固定画面或ID 1)
            id_top = sorted_ids[1] if len(sorted_ids) > 1 else None

            # === 智能比例分配 ===
            if id_top is not None:
                # 场景：两人对谈 -> 50% : 50%
                split_ratio = 0.5
                logger.info(f"双主体分屏: Bottom(ID={id_bottom}), Top(ID={id_top})")
            else:
                # 场景：单人 + 背景/PPT -> 65% : 35% (人更大)
                split_ratio = 0.65
                logger.info(f"单主体分屏: Bottom(ID={id_bottom})")

            # 计算具体高度
            h_bottom = int(target_h * split_ratio)
            h_top = target_h - h_bottom

            # 找到 id_bottom 第一次出现的坐标
            start_pos_bottom = self._get_first_position(id_bottom, raw_tracks_history, src_w, src_h)
            logger.info(f"Bottom主体第一次出现的坐标: {start_pos_bottom}")

            # 找到 id_top 第一次出现的坐标 (如果不存在则用计算逻辑)
            if id_top:
                start_pos_top = self._get_first_position(id_top, raw_tracks_history, src_w, src_h)
                logger.info(f"Top主体第一次出现的坐标: {start_pos_top}")
            else:
                # 单人模式背景逻辑：如果主体在左，背景初始在右
                if start_pos_bottom[0] < src_w / 2:
                    start_pos_top = (src_w * 0.75, src_h / 2)
                else:
                    start_pos_top = (src_w * 0.25, src_h / 2)

            # --- 使用正确位置初始化滤波器 ---
            preset = SMOOTHING_PRESETS.get(mode, SMOOTHING_PRESETS["stable"])

            # Bottom Filter
            self.smoother_x = OneEuroFilter(t0=0, x0=start_pos_bottom[0], min_cutoff=preset["min_cutoff"],
                                            beta=preset["beta"])
            self.smoother_y = OneEuroFilter(t0=0, x0=start_pos_bottom[1], min_cutoff=preset["min_cutoff"],
                                            beta=preset["beta"])

            # Top Filter (现在参数和 Bottom 保持一致，更稳定)
            smoother_top_x = OneEuroFilter(t0=0, x0=start_pos_top[0], min_cutoff=preset["min_cutoff"],
                                           beta=preset["beta"])
            smoother_top_y = OneEuroFilter(t0=0, x0=start_pos_top[1], min_cutoff=preset["min_cutoff"],
                                           beta=preset["beta"])

            path_bottom = []
            path_top = []

            # 记录上一次的有效位置 (补偿逻辑主体丢失的问题)
            last_valid_bottom = (src_w / 2, src_h / 2)
            last_valid_top = (src_w / 2, src_h / 2)

            for i in range(total_frames):
                ts = i / fps

                # --- 计算下半屏 (Main) ---
                if i in raw_tracks_history[id_bottom]:
                    bbox = raw_tracks_history[id_bottom][i]
                    tx, ty = self._calc_smart_center(bbox)
                    last_valid_bottom = (tx, ty)  # 更新记忆
                else:
                    # 丢失时，使用最后一次已知位置 (Hold)，而不是回中
                    tx, ty = last_valid_bottom

                sx = self.smoother_x(ts, tx)
                sy = self.smoother_y(ts, ty)
                # 注意：计算 crop 时用 h_bottom
                path_bottom.append(self._calc_crop_xy(sx, sy, target_w, h_bottom, src_w, src_h))

                # --- 计算上半屏 (Secondary) ---
                if id_top and i in raw_tracks_history[id_top]:
                    bbox = raw_tracks_history[id_top][i]
                    tx2, ty2 = self._calc_smart_center(bbox)
                    last_valid_top = (tx2, ty2)
                elif id_top:
                    tx2, ty2 = last_valid_top
                else:
                    # 单人背景逻辑 (反向跟随)
                    if sx < src_w / 2:
                        tx2 = src_w * 0.75
                    else:
                        tx2 = src_w * 0.25
                    ty2 = src_h / 2

                sx2 = smoother_top_x(ts, tx2)
                sy2 = smoother_top_y(ts, ty2)
                # 注意传入 h_top
                path_top.append(self._calc_crop_xy(sx2, sy2, target_w, h_top, src_w, src_h))

            # 渲染分屏
            self._render_split_screen_cv2(input_path, output_path, path_top, path_bottom, target_w, target_h, h_top, h_bottom, fps)
            return [output_path]
        else:
            # --- 不开分屏模式 ---
            logger.info(f"分别导出 {len(valid_ids)} 个视频")
            generated = []
            for idx, tid in enumerate(valid_ids):
                logger.info(f"正在处理第 {idx + 1}/{len(valid_ids)} 个主体 (ID: {tid})")

                # 为每个 ID 初始化独立的滤波器
                self._init_smoothers(src_w, src_h, mode)

                # 获取该 ID 的初始位置 (防止从画面中心飘过去)
                start_pos = self._get_first_position(tid, raw_tracks_history, src_w, src_h)

                # 生成该 ID 的专属路径
                id_path = []

                # 初始化记忆位置
                last_valid = start_pos

                for i in range(total_frames):
                    if i in raw_tracks_history[tid]:
                        bbox = raw_tracks_history[tid][i]
                        if tid == 2:
                            logger.info(f"当前主体【{tid}】的bbox={bbox}")
                        tx, ty = self._calc_smart_center(bbox)
                        last_valid = (tx, ty)
                    else:
                        tx, ty = last_valid

                    # 滤波
                    sx = self.smoother_x(i / fps, tx)
                    sy = self.smoother_y(i / fps, ty)
                    id_path.append(self._calc_crop_xy(sx, sy, target_w, target_h, src_w, src_h))

                base, ext = os.path.splitext(output_path)
                sub_out = f"{base}_subject_{tid}{ext}"
                self._render_with_ffmpeg(input_path, sub_out, id_path, target_w, target_h, fps)
                generated.append(sub_out)
            return generated

    def _get_first_position(self, tid, history, src_w, src_h):
        """
        获取某个 ID 第一次出现的智能中心坐标。
        tid: 目标 ID
        history: 完整的轨迹字典 {id: {frame: bbox}}
        """
        # 1. 安全检查：确保 ID 存在
        if tid not in history:
            return (src_w / 2, src_h / 2)

        # 2. 获取该 ID 的专属轨迹 {frame: bbox}
        id_history = history[tid]

        # 3. 对帧号排序，找到第一帧
        frames = sorted(id_history.keys())
        if not frames:
            return (src_w / 2, src_h / 2)

        first_frame = frames[0]
        bbox = id_history[first_frame]

        # 4. 计算那一帧的智能重心
        return self._calc_smart_center(bbox)

    def _calc_target_size(self, src_w, src_h, ratio_str):
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

        # 如果是奇数，减 1 变成偶数
        if target_w % 2 != 0: target_w -= 1
        if target_h % 2 != 0: target_h -= 1

        return target_w, target_h


    def _init_smoothers(self, src_w, src_h, mode):
        preset = SMOOTHING_PRESETS.get(mode, SMOOTHING_PRESETS["normal"])
        logger.info(f"使用平滑参数: {preset}")
        self.smoother_x = OneEuroFilter(t0=0, x0=src_w / 2, min_cutoff=preset["min_cutoff"], beta=preset["beta"])
        self.smoother_y = OneEuroFilter(t0=0, x0=src_h / 2, min_cutoff=preset["min_cutoff"], beta=preset["beta"])

    def _calc_crop_xy(self, sx, sy, tw, th, sw, sh):
        crop_x = sx - (tw / 2)
        crop_x = max(0, min(crop_x, sw - tw))
        crop_y = sy - (th / 2)
        crop_y = max(0, min(crop_y, sh - th))
        return (int(crop_x), int(crop_y))

    # --- OpenCV 拼接渲染器 (解决 FFmpeg 复杂指令问题) ---
    def _render_split_screen_cv2(self, input_path, output_path, path_top, path_bottom, w, h, h_top, h_bottom, fps):
        # 使用稳定的 CPU 编码
        video_encoder = 'libx264'

        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',  # 总尺寸
            '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-',
            '-i', input_path,
            '-map', '0:v', '-map', '1:a?',
            '-c:v', video_encoder, '-preset', 'veryfast', '-crf', '23',
            '-vf', 'format=yuv420p',
            '-c:a', 'aac', '-b:a', '192k', '-shortest',
            output_path
        ]

        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=10 ** 7)
        cap = cv2.VideoCapture(input_path)
        frame_idx = 0

        # 创建分割线颜色 (白色)
        separator_color = (255, 255, 255)
        separator_thickness = 2

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_idx >= len(path_top): break

            # 1. 裁剪 Bottom (高度 h_bottom)
            bx, by = path_bottom[frame_idx]
            img_bottom = self._safe_crop(frame, bx, by, w, h_bottom)

            # 2. 裁剪 Top (高度 h_top)
            tx, ty = path_top[frame_idx]
            img_top = self._safe_crop(frame, tx, ty, w, h_top)

            # 3. 拼接
            try:
                # 可选：在 img_top 底部画一条线，或者直接拼
                final_frame = cv2.vconcat([img_top, img_bottom])
                process.stdin.write(final_frame.tobytes())
            except Exception as e:
                logger.error(f"Stitch error: {e}")
                break

            frame_idx += 1

        cap.release()
        process.stdin.close()
        process.wait()

    def _safe_crop(self, frame, x, y, w, h):
        """防止裁剪越界的 helper"""
        max_h, max_w = frame.shape[:2]
        x = max(0, min(x, max_w - w))
        y = max(0, min(y, max_h - h))
        return frame[y:y + h, x:x + w]

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
            '-map', '1:a?',  # 只要音频流 如果没音频就不复制
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
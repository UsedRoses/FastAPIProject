# 剪辑主逻辑 (Video Processing Pipeline)

import cv2
import subprocess
import os
import numpy as np
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

# 死区阈值
DEAD_ZOOM_PRESETS = {
    "fast": 10,
    "normal": 40,
    "stable": 80,
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

        # --- 根据形态决定重心 ---
        aspect_ratio = w / h

        if aspect_ratio < 0.8:
            # 瘦高型 (站立的人): 重心上移 (瞄准胸部/头部)
            # 0.35 表示从顶部落下 35% 的距离
            smart_center_y = y1 + (h * 0.35)
        elif aspect_ratio > 1.2:
            # 扁平型 (趴着的猫、车): 重心居中，甚至稍微偏下以展示地面
            # 0.5 表示几何中心
            smart_center_y = y1 + (h * 0.5)
        else:
            # 方形 (坐着的人/大头照): 稍微上移一点点
            smart_center_y = y1 + (h * 0.45)

        return center_x, smart_center_y

    def _calc_auto_zoom(self, bbox, target_w, target_h, src_w, src_h):
        """计算为了包住主体所需的 Zoom Out 系数"""
        x1, y1, x2, y2 = bbox
        subj_w = x2 - x1

        # 留白 15%
        padding = 1.15
        needed_w = subj_w * padding

        # 如果主体宽度 > 目标宽度，需要 Zoom > 1.0 (缩小画面)
        # 否则 Zoom = 1.0 (保持原画质，不进行数码变焦放大，防止模糊)
        zoom = needed_w / target_w

        # 限制：最大不能超过原视频宽度
        max_zoom = src_w / target_w
        return max(1.0, min(zoom, max_zoom))

    # --- 智能布局计算引擎 ---
    def _get_smart_layout(self, num_subjects, target_w, target_h):
        """
        根据主体数量，返回每个主体在画布上的 (x, y, w, h)
        返回格式: [(x, y, w, h), (x, y, w, h), ...]
        列表索引 0 对应最主要的主体 (ID出现时间最长的)
        """
        layout_slots = []

        if num_subjects == 1:
            # 全屏
            layout_slots.append((0, 0, target_w, target_h))

        elif num_subjects == 2:
            # 上下平分
            h_half = target_h // 2
            layout_slots.append((0, h_half, target_w, target_h - h_half))  # 主体1放下面(通常是主视角)
            layout_slots.append((0, 0, target_w, h_half))  # 主体2放上面

        elif num_subjects == 3:
            # 品字形布局 (上面1个大的，下面2个小的)
            # 这种布局适合竖屏，如果是横屏输出，可能需要左右结构
            # 假设输出是 9:16

            # Top: 上半部分，高度 50%
            h_top = target_h // 2
            h_bottom = target_h - h_top

            # Top Slot (大图): 放 ID 1
            layout_slots.append((0, 0, target_w, h_top))

            # Bottom Slots (小图): 左右平分
            w_half = target_w // 2
            layout_slots.append((0, h_top, w_half, h_bottom))  # 左下
            layout_slots.append((w_half, h_top, target_w - w_half, h_bottom))  # 右下

        elif num_subjects == 4:
            # 田字格 (2x2)
            w_half = target_w // 2
            h_half = target_h // 2

            layout_slots.append((0, 0, w_half, h_half))  # 左上
            layout_slots.append((w_half, 0, w_half, h_half))  # 右上
            layout_slots.append((0, h_half, w_half, h_half))  # 左下
            layout_slots.append((w_half, h_half, w_half, h_half))  # 右下

        else:
            # > 4 的情况，简单暴力处理：全部做成横条 (或者你可以扩展 3x2)
            h_per = target_h // num_subjects
            for i in range(num_subjects):
                y = i * h_per
                h = h_per if i < num_subjects - 1 else (target_h - y)
                layout_slots.append((0, y, target_w, h))

        return layout_slots

    def process_video(self,
                      input_path,
                      output_path,
                      ratio_str="9:16",
                      mode="normal",
                      detect_target="person",
                      multi_subject=False,
                      split_screen=False,
                      max_split=4,
                      debug=False
                      ):
        """
        核心流程入口
        ratio_str: "9:16", "4:3", "1:1", "16:9" 等字符串
        mode: "fast", "normal", "stable"

        根据 multi_subject 决定走单人模式还是多人模式
        """
        self.tracker = VideoTracker()

        logger.info(f"开始处理视频: {input_path} | 比例: {ratio_str} | 模式: {mode} | 目标: {detect_target} | 分镜拆条: {multi_subject} | 分屏: {split_screen}")

        if debug:
            return self.run_debug(input_path, output_path, detect_target)

        src_w, src_h, fps, total_frames = self._get_video_info(input_path)
        target_w, target_h = self._calc_target_size(src_w, src_h, ratio_str)
        target_ids = get_ids_by_names(detect_target)

        # 全量扫描 (Scan Phase)
        raw_tracks_history = defaultdict(dict)
        cap = cv2.VideoCapture(input_path)
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            detections = self.detector.detect(frame, target_ids=target_ids)
            tracks = self.tracker.update(detections, (src_h, src_w))
            for t in tracks:
                # 记录 bbox [x1, y1, x2, y2]
                raw_tracks_history[t['id']][i] = t['bbox']
            if i % 100 == 0: logger.info(f"Scanning: {i}/{total_frames}")
        cap.release()

        # 筛选有效主体 (Filter Phase)
        # 只要出现超过 1.5秒 (45帧) 就算有效，防止漏掉
        absolute_min_frames = 45
        sorted_candidates = sorted(
            [(tid, len(h)) for tid, h in raw_tracks_history.items() if len(h) > absolute_min_frames],
            key=lambda x: x[1],
            reverse=True
        )
        valid_ids = [x[0] for x in sorted_candidates]
        logger.info(f"检测到的主体(ID/帧数): {sorted_candidates}")

        # 如果没找到任何主体，这就变成了一个纯风景视频，我们可以虚拟一个 ID=None 的任务
        if not valid_ids:
            logger.warning("无有效主体，进入默认背景模式")
            valid_ids = [None]

        # 路径生成与渲染 (Path Generation & Render)

        # === A. 分屏合并模式 (Split Screen) ===
        if split_screen and multi_subject:
            # 截取前 N 个
            selected_ids = valid_ids[:max_split]
            num_subjects = len(selected_ids)
            logger.info(f"应用 {num_subjects} 分屏布局")

            # --- 获取布局配置 ---
            # slots: [(x, y, w, h), ...] 对应 selected_ids 的顺序
            layout_slots = self._get_smart_layout(num_subjects, target_w, target_h)

            # 并行生成所有路径
            all_paths = []
            for i, tid in enumerate(selected_ids):
                # 获取该 ID 在布局中被分配的宽高
                slot_x, slot_y, slot_w, slot_h = layout_slots[i]

                # 生成路径 (注意：传入的是格子的尺寸，自动变焦会根据格子大小调整)
                path = self._generate_track_path(
                    tid, raw_tracks_history, total_frames, fps,
                    src_w, src_h, slot_w, slot_h, mode
                )
                all_paths.append(path)

            # 渲染网格
            self._render_grid_cv2(
                input_path, output_path, all_paths, layout_slots, target_w, target_h, fps
            )
            return [output_path]

        # === B. 独立导出模式 (Single / Multi Separate) ===
        else:
            # 如果不是多主体模式，只取第一个 ID
            export_ids = valid_ids if multi_subject else [valid_ids[0]]

            logger.info(f"独立导出模式，将生成 {len(export_ids)} 个视频")
            generated = []

            for tid in export_ids:
                # 生成路径
                path = self._generate_track_path(tid, raw_tracks_history, total_frames, fps, src_w, src_h, target_w, target_h, mode)

                # 构造文件名
                sub_out = output_path
                if len(export_ids) > 1 and tid is not None:
                    base, ext = os.path.splitext(output_path)
                    sub_out = f"{base}_subject_{tid}{ext}"

                # 渲染 (单屏其实就是 1 层的堆叠)
                layout_slots = [(0, 0, target_w, target_h)]
                self._render_grid_cv2(input_path, sub_out, [path], layout_slots, target_w, target_h, fps)
                generated.append(sub_out)

            return generated

    # ==================== 核心逻辑 通用路径生成器 ====================
    def _generate_track_path(self, tid, raw_tracks_history, total_frames, fps, src_w, src_h, target_w, target_h,
                             mode):
        """
        为指定 ID 生成一条 (x, y, zoom) 的完整路径。
        包含了：智能重心、Hold丢失补偿、自动变焦、OneEuro滤波。
        """
        # 1. 确定初始位置
        history = raw_tracks_history.get(tid, {}) if tid is not None else {}

        if history:
            # 找到第一帧
            first_frame = sorted(history.keys())[0]
            start_bbox = history[first_frame]
            start_cx, start_cy = self._calc_smart_center(start_bbox)
            start_zoom = self._calc_auto_zoom(start_bbox, target_w, target_h, src_w, src_h)
        else:
            # 如果没有历史 (ID=None 或 空)，居中
            start_cx, start_cy = src_w / 2, src_h / 2
            start_zoom = 1.0

        # 2. 初始化独立滤波器
        preset = SMOOTHING_PRESETS.get(mode, SMOOTHING_PRESETS["normal"])
        smoother_x = OneEuroFilter(t0=0, x0=start_cx, min_cutoff=preset["min_cutoff"], beta=preset["beta"])
        smoother_y = OneEuroFilter(t0=0, x0=start_cy, min_cutoff=preset["min_cutoff"], beta=preset["beta"])
        # Zoom 变化要慢一点，防止晕车
        smoother_z = OneEuroFilter(t0=0, x0=start_zoom, min_cutoff=0.01, beta=0.005)

        path = []
        # --- 状态变量 ---
        # 用于处理丢失情况
        last_valid_pos = (start_cx, start_cy)
        last_valid_zoom = start_zoom

        # --- 新增：用于死区逻辑的状态变量 ---
        # 记录上一次“真正移动了镜头”时的目标位置
        last_fed_x = start_cx
        last_fed_y = start_cy

        # 死区阈值 N
        # 意思：如果检测到的中心点变化小于 50 像素，就当它没动
        # 对于访谈类 (stable)，建议设大一点 (如 50-80)
        # 对于运动类 (fast)，建议设小一点 (如 10-20)
        dead_zone_n = DEAD_ZOOM_PRESETS.get(mode, DEAD_ZOOM_PRESETS["normal"])

        for i in range(total_frames):
            ts = i / fps

            if i in history:
                bbox = history[i]
                tx, ty = self._calc_smart_center(bbox)
                tz = self._calc_auto_zoom(bbox, target_w, target_h, src_w, src_h)

                last_valid_pos = (tx, ty)
                last_valid_zoom = tz
            else:
                # 丢失时，保持最后状态 (Hold)
                tx, ty = last_valid_pos
                tz = last_valid_zoom

            # 只有当新位置 (tx) 和上一次锁定的位置 (last_fed_x) 差距超过 N 时，才更新目标
            # 否则，强行把目标按住在原地

            # X轴判断
            if abs(tx - last_fed_x) < dead_zone_n:
                tx = last_fed_x  # 没超过阈值，欺骗滤波器说“我没动”
            else:
                last_fed_x = tx  # 超过了，更新锁定位置

            # Y轴判断
            if abs(ty - last_fed_y) < dead_zone_n:
                ty = last_fed_y
            else:
                last_fed_y = ty

            # 3D 滤波
            sx = smoother_x(ts, tx)
            sy = smoother_y(ts, ty)
            sz = smoother_z(ts, tz)

            path.append((int(sx), int(sy), sz))

        return path

    # --- 通用网格渲染器 (支持任意排版) ---
    def _render_grid_cv2(self, input_path, output_path, paths_list, layout_slots, final_w, final_h, fps):
        """
        paths_list: [path_subj_1, path_subj_2, ...]
        layout_slots: [(x,y,w,h), (x,y,w,h), ...] 对应每个主体在画布的位置
        """
        video_encoder = 'libx264'
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{final_w}x{final_h}',
            '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-', '-i', input_path,
            '-map', '0:v', '-map', '1:a?',
            '-c:v', video_encoder, '-preset', 'veryfast', '-crf', '23',
            '-vf', 'format=yuv420p',
            '-c:a', 'aac', '-b:a', '192k', '-shortest',
            output_path
        ]

        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=10 ** 7)
        cap = cv2.VideoCapture(input_path)
        frame_idx = 0

        # 预创建一个黑色画布 (避免每帧都 create，提升性能)
        # 注意：这里不能预创建，因为每帧都要清空或者覆盖。但我们可以创建一个 base_canvas。

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_idx >= len(paths_list[0]): break

            # 创建黑色背景画布
            final_canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)

            # 遍历每个主体，贴到对应的 slot 里
            for i, path in enumerate(paths_list):
                # 获取该主体的目标格子信息
                slot_x, slot_y, slot_w, slot_h = layout_slots[i]

                # 获取追踪参数
                cx, cy, zoom = path[frame_idx]

                # --- 核心裁剪逻辑 (复用之前 Zoom + Letterbox) ---
                real_w = int(slot_w * zoom)
                real_h = int(slot_h * zoom)

                crop_x = int(cx - real_w / 2)
                crop_y = int(cy - real_h / 2)

                raw_crop = self._safe_crop(frame, crop_x, crop_y, real_w, real_h)

                # Resize 到格子大小
                # 保持宽度撑满格子 (slot_w)，高度自适应
                scale = slot_w / real_w
                resized_h = int(raw_crop.shape[0] * scale)
                img_resized = cv2.resize(raw_crop, (slot_w, resized_h), interpolation=cv2.INTER_LINEAR)

                # Letterbox 处理 (如果 resize 后高度 < slot_h，居中补黑；如果 > slot_h，裁中间)
                y_offset = (slot_h - resized_h) // 2

                if y_offset >= 0:
                    # 放入格子中心
                    final_canvas[slot_y + y_offset: slot_y + y_offset + resized_h,
                    slot_x: slot_x + slot_w] = img_resized
                else:
                    # 裁剪图片中心
                    y_start = -y_offset
                    final_canvas[slot_y: slot_y + slot_h, slot_x: slot_x + slot_w] = \
                        img_resized[y_start: y_start + slot_h, :]

                # 可选：绘制分割线 (比如在格子边缘画白线)
                # cv2.rectangle(final_canvas, (slot_x, slot_y), (slot_x+slot_w, slot_y+slot_h), (255,255,255), 2)

            try:
                process.stdin.write(final_canvas.tobytes())
            except Exception as e:
                logger.error(f"Render Error: {e}")
                break

            frame_idx += 1

        cap.release()
        process.stdin.close()
        process.wait()

    def run_debug(self,
                  input_path,
                  output_path,
                  detect_target="person",
                  ):
        src_w, src_h, fps, total_frames = self._get_video_info(input_path)
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

        debug_filename = output_path.replace("reframed_", "debug_")
        self._render_debug_video(input_path, debug_filename, raw_tracks_history, fps)
        # 调试模式下，生成完画框视频直接返回，不进行剪辑
        return [debug_filename]

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
        """增强版安全裁剪：越界自动补黑边"""
        img_h, img_w = frame.shape[:2]

        # 如果完全在范围内，直接切 (最快)
        if x >= 0 and y >= 0 and x + w <= img_w and y + h <= img_h:
            return frame[y:y + h, x:x + w]

        # 否则创建画布
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # 计算重叠区域
        src_x1 = max(0, x)
        src_y1 = max(0, y)
        src_x2 = min(img_w, x + w)
        src_y2 = min(img_h, y + h)

        dst_x1 = max(0, -x)
        dst_y1 = max(0, -y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

        return canvas

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

    def _render_debug_video(self, input_path, output_path, raw_tracks_history, fps):
        """
        生成调试视频：画出所有检测到的框和 ID
        """
        logger.info("正在生成调试视频 (Debug Visualization)...")
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 使用 CPU 编码器生成画框视频
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '25',
            '-pix_fmt', 'yuv420p',
            output_path
        ]

        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=10 ** 7)
        frame_idx = 0

        # 颜色表 (用于区分不同 ID)
        colors = np.random.randint(0, 255, (100, 3)).tolist()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 画框
            # 遍历所有 ID，看这一帧谁在
            for tid, history in raw_tracks_history.items():
                if frame_idx in history:
                    bbox = history[frame_idx]
                    x1, y1, x2, y2 = map(int, bbox)
                    color = colors[tid % len(colors)]

                    # 画矩形
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                    # 写 ID
                    cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            try:
                process.stdin.write(frame.tobytes())
            except:
                break
            frame_idx += 1

        cap.release()
        process.stdin.close()
        process.wait()
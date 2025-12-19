# 封装 ByteTrack (Tracking)
from app.core.byte_tracker import BYTETracker
from dataclasses import dataclass


@dataclass
class TrackerConfig:
    track_thresh: float = 0.5  # 只有置信度大于 0.5 的框才参与匹配
    track_buffer: int = 30  # 如果一个人消失了，保留 30 帧的记忆
    match_thresh: float = 0.8  # 匹配相似度
    mot20: bool = False


class VideoTracker:
    def __init__(self, fps=30):
        args = TrackerConfig()
        self.tracker = BYTETracker(args, frame_rate=fps)

    def update(self, detections, img_size):
        if detections is None or len(detections) == 0:
            return []

        online_targets = self.tracker.update(detections, img_size, (img_size[0], img_size[1]))

        results = []
        for t in online_targets:
            # --- 核心修改：使用 .tlbr 而不是 .tlwh ---
            # tlbr = [x1, y1, x2, y2] (Top-Left, Bottom-Right)
            # 这样所有的后续逻辑都不用动了
            tlbr = t.tlbr

            # center 计算依然没问题
            center_x = (tlbr[0] + tlbr[2]) / 2
            center_y = (tlbr[1] + tlbr[3]) / 2

            results.append({
                "id": t.track_id,
                "bbox": tlbr,  # <--- 修正为 tlbr
                "center": (center_x, center_y),
                "score": t.score
            })
        return results
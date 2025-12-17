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
        """
        detections: np.array [[x1, y1, x2, y2, score], ...]
        img_size: (height, width)
        """
        # ByteTracker 需要的输入格式是 Tensor 或者特定的 numpy 结构
        # 这里直接传 numpy 即可，YOLOX 的实现里处理了
        if detections is None or len(detections) == 0:
            return []

        # update 返回的是 STrack 对象列表
        online_targets = self.tracker.update(detections, img_size, (img_size[0], img_size[1]))

        results = []
        for t in online_targets:
            tlwh = t.tlwh  # top-left width height
            tid = t.track_id
            # 转换成中心点，方便后续计算
            center_x = tlwh[0] + tlwh[2] / 2
            center_y = tlwh[1] + tlwh[3] / 2
            results.append({
                "id": tid,
                "bbox": tlwh,
                "center": (center_x, center_y),
                "score": t.score
            })
        return results
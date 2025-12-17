import numpy as np
from collections import deque
import scipy.linalg


class KalmanFilter:
    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T
        )) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        ))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T
        ))
        return new_mean, new_covariance


def iou_batch(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray
    :rtype iou_matrix np.ndarray
    """
    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float64)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float64)

    an = len(atlbrs)
    bn = len(btlbrs)

    # areas
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    area_a = box_area(atlbrs)
    area_b = box_area(btlbrs)

    # intersections
    top_left = np.maximum(atlbrs[:, None, :2], btlbrs[None, :, :2])
    bottom_right = np.minimum(atlbrs[:, None, 2:], btlbrs[None, :, 2:])
    area_inter = np.prod(np.maximum(bottom_right - top_left, 0), axis=2)

    return area_inter / (area_a[:, None] + area_b[None, :] - area_inter)


class STrack:
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score):
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != 1:  # TrackState.Tracked
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = 1  # TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = 1  # TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = 1  # TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def next_id():
        if not hasattr(STrack, '_count'):
            STrack._count = 0
        STrack._count += 1
        return STrack._count


class BYTETracker:
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

        self.frame_id = 0
        self.args = args
        self.det_thresh = args.track_thresh
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 格式化检测框: x1, y1, x2, y2, score
        if len(output_results) > 0:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]  # x1y1x2y2
        else:
            scores = np.array([])
            bboxes = np.array([])

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)

        # High score detections
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        detections = [STrack(self.tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(dets, scores_keep)]

        # Low score detections
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]
        detections_second = [STrack(self.tlbr_to_tlwh(tlbr), s) for tlbr, s in zip(dets_second, scores_second)]

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = join_stracks(tracked_stracks, self.lost_stracks)

        # Predict current location using Kalman filter
        for track in strack_pool:
            track.predict()

        # Association with high score detections
        dists = self.iou_distance(strack_pool, detections)
        matches, u_track, u_detection = self.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == 1:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Association with low score detections
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == 1]
        dists = self.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = self.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == 1:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == 2:  # TrackState.Lost
                track.mark_lost()
                lost_stracks.append(track)

        # Deal with unconfirmed tracks
        detections = [detections[i] for i in u_detection]
        dists = self.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = self.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Init new tracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        # Update state
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 1]
        self.tracked_stracks = join_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = join_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        return [t for t in self.tracked_stracks if t.is_activated]

    def linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

        try:
            import lap
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
            matches = [[ix, iy] for ix, iy in enumerate(x) if iy >= 0]
            unmatched_a = np.where(x < 0)[0]
            unmatched_b = np.where(y < 0)[0]
        except ImportError:
            from scipy.optimize import linear_sum_assignment
            x, y = linear_sum_assignment(cost_matrix)
            matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
            if len(matches) == 0:
                unmatched_a = list(range(cost_matrix.shape[0]))
                unmatched_b = list(range(cost_matrix.shape[1]))
            else:
                unmatched_a = list(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
                unmatched_b = list(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

        return matches, unmatched_a, unmatched_b

    def iou_distance(self, atracks, btracks):
        """
        计算两组轨迹之间的 IoU 距离
        """
        # 1. 提取坐标 (tlbr: top-left-bottom-right)
        if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
                (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
            atlbrs = atracks
            btlbrs = btracks
        else:
            atlbrs = [track.tlbr for track in atracks]
            btlbrs = [track.tlbr for track in btracks]

        # --- 核心修复开始 ---
        # 务必先转换为 numpy 数组
        atlbrs = np.array(atlbrs)
        btlbrs = np.array(btlbrs)

        # 关键逻辑：如果数组为空，强制 Reshape 为 (0, 4)
        # 这样 NumPy 就能正确处理 [:, 2] 这种切片操作，而不会报错
        if len(atlbrs) == 0:
            atlbrs = atlbrs.reshape(0, 4)
        if len(btlbrs) == 0:
            btlbrs = btlbrs.reshape(0, 4)
        # --- 核心修复结束 ---

        _ious = iou_batch(atlbrs, btlbrs)
        cost_matrix = 1 - _ious
        return cost_matrix

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret


# 辅助函数
def join_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_batch(np.array([t.tlbr for t in stracksa]), np.array([t.tlbr for t in stracksb]))
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


# 增加缺失的方法，防止报错
setattr(STrack, 'mark_lost', lambda self: setattr(self, 'state', 2))  # 2=Lost
setattr(STrack, 'mark_removed', lambda self: setattr(self, 'state', 3))  # 3=Removed
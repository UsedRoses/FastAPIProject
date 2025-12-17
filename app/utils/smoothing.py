# One-Euro Filter 平滑算法
import math
import time


class OneEuroFilter:
    def __init__(self, t0=0, x0=0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        初始化一欧元滤波器 (One Euro Filter)

        参数:
        min_cutoff: 最小截止频率 (Hz)。
                    越小越平滑（延迟越高），越大越灵敏（抖动越多）。
                    推荐值: 0.01 ~ 1.0
        beta:       速度系数。
                    当目标快速移动时，截止频率会增加以减少延迟。
                    推荐值: 0.001 ~ 0.1
        d_cutoff:   导数的截止频率 (Hz)，通常设为 1.0 即可。
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        """
        滤波函数

        参数:
        t: 当前时间戳 (frame_index / fps) 或 (time.time())
        x: 当前测量的原始值 (例如 detected_center_x)

        返回:
        平滑后的值
        """
        t_e = t - self.t_prev

        # 避免时间戳重复或倒流导致的除零错误
        if t_e <= 0:
            return self.x_prev

        # 计算信号变化率 (速度 dx)
        # 对速度也进行低通滤波，防止速度突变
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # 动态计算截止频率 (Adaptive Cutoff)
        # 速度越快，cutoff 越高（越灵敏）；速度越慢，cutoff 越低（越平滑）
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # 对位置进行滤波
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # 更新历史状态
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
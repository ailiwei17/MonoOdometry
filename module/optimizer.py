import numpy as np
from scipy.optimize import least_squares


class BundleAdjustmentOptimer(object):
    def __init__(self, now_pose, points4d, observe_kps):
        self.points4d = points4d
        self.points3d = points4d[:, :3]
        now_pose = np.linalg.inv(now_pose)
        self.R = now_pose[:3, :3]
        self.t = now_pose[:3, 3]
        self.observe_kps = observe_kps

    def project_points(self, params):
        # 转换到相机坐标系
        points_cam = np.dot(params[0:9].reshape(3, 3), self.points3d.T) + params[9:].reshape(3, 1)
        # 转换到像素坐标系
        points_pixel = points_cam[:2] / points_cam[2]
        return points_pixel.T

    def reprojection_error(self, params):
        reprojections = self.project_points(params)
        errors = np.ravel(reprojections - self.observe_kps)
        print(errors)
        return np.sum(errors ** 2)

    def objective_func(self, params):
        return self.reprojection_error(params)

    def update(self):
        """
        局部BA优化：
        求第二帧的像素坐标系特征点
        最小化和观测的第二帧的像素坐标系特征点误差
        解pose
        """
        init_params = np.concatenate([self.R.ravel(), self.t.ravel()])
        result = least_squares(self.objective_func, init_params)
        optimized_params = result.x
        R = optimized_params[:9].reshape((3, 3))
        t = optimized_params[9:]
        now_pose = np.eye(4)
        now_pose[:3, :3] = R
        now_pose[:3, 3] = t.flatten()
        return now_pose

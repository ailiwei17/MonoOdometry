import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform


class Frame(object):
    idx = 0
    last_kps, last_des, last_pose, last_Rt = None, None, None, np.eye(4)

    def __init__(self, image, k):
        """把上一帧的信息传递给下一帧：通过类属性"""
        Frame.idx += 1
        self.image = image
        self.idx = Frame.idx
        self.last_kps = Frame.last_kps
        self.last_des = Frame.last_des
        self.last_pose = Frame.last_pose
        self.last_Rt = Frame.last_Rt
        self.now_kps, self.now_des, self.now_pose, self.now_Rt = None, None, None, np.eye(4)
        self.norm_now_kps, self.norm_last_kps = None, None
        self.k = k

    def process_frame(self):
        """处理图像"""
        self.now_kps, self.now_des = self.extract_points()
        Frame.last_kps, Frame.last_des, Frame.last_Rt = self.now_kps, self.now_des, self.now_Rt
        # 初始化
        if self.idx == 1:
            self.now_pose = np.eye(4)
        else:
            # 寻找匹配上的关键点
            match_kps = self.match_points()
            # 拟合本质矩阵
            essential_matrix = self.fit_essential_matrix(match_kps)
            """
            利用本质矩阵分解出相机的位姿
            通过本质矩阵分解，可以得到两组可能的旋转矩阵和平移向量
            用归一化的点最小重投影误差
            """
            _, R, t, _ = cv2.recoverPose(essential_matrix, self.norm_now_kps, self.norm_last_kps)
            self.now_Rt[:3, :3] = R
            self.now_Rt[:3, 3] = t.flatten()
            # 计算当前帧相当于上一帧的位姿变化
            self.now_pose = np.dot(self.now_Rt, self.last_pose)
            # 三角测量得到空间位置
            points4d = self.triangulate()
            good_pt4d = self.check_points(points4d)
            self.last_kps = self.last_kps[good_pt4d]
            self.now_kps = self.now_kps[good_pt4d]
            self.draw_points()
        Frame.last_pose = self.now_pose
        Frame.last_Rt = self.now_Rt
        return self.image, self.now_pose

    def extract_points(self):
        """提取角点"""
        orb = cv2.ORB_create()
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # 从图像中提取关键点,pts的shape为(N,1,2)
        pts = cv2.goodFeaturesToTrack(image, 3000, qualityLevel=0.01, minDistance=0.1)
        # 用列表推导式转换格式
        kps = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=20) for pt in pts]
        # kps: 更新后的关键点列表, des: 计算得到的特征描述子矩阵，用于描述每个关键点的局部图像特征
        kps, des = orb.compute(image, kps)
        kps = np.array([(kp.pt[0], kp.pt[1]) for kp in kps])
        return kps, des

    def match_points(self):
        # 暴力匹配
        bfmatch = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # k=2 表示匹配每个查询描述子时返回的最佳匹配和次好匹配
        matches = bfmatch.knnMatch(self.now_des, self.last_des, k=2)
        match_kps, idx1, idx2 = [], [], []
        for m, n in matches:
            # 将最佳匹配与次好匹配作比较，判断最佳匹配是否可用
            if m.distance < 0.75 * n.distance:
                # 将最佳匹配的查询索引（当前帧的特征点索引）添加到 idx1 列表中
                idx1.append(m.queryIdx)
                # 将最佳匹配的训练索引（上一帧的特征点索引）添加到 idx2 列表中
                idx2.append(m.trainIdx)
                p1 = self.now_kps[m.queryIdx]
                p2 = self.last_kps[m.trainIdx]
                match_kps.append((p1, p2))
        # 丢失匹配
        assert len(match_kps) >= 8

        self.now_kps = self.now_kps[idx1]
        self.last_kps = self.last_kps[idx2]
        return match_kps

    def fit_essential_matrix(self, match_kps):
        match_kps = np.array(match_kps)

        # 使用相机内参对角点坐标归一化
        self.norm_now_kps = self.normalize(self.k, match_kps[:, 0])
        self.norm_last_kps = self.normalize(self.k, match_kps[:, 1])

        # 求解本质矩阵和内点数据，通过迭代多次随机采样和模型拟合，找到拟合误差最小的模型，即最优的本质矩阵
        model, inliers = ransac((self.norm_last_kps, self.norm_now_kps),
                                EssentialMatrixTransform,
                                min_samples=10,
                                residual_threshold=0.005,
                                max_trials=200)
        self.now_kps = self.now_kps[inliers]
        self.last_kps = self.last_kps[inliers]
        self.norm_now_kps = self.norm_now_kps[inliers]
        self.norm_last_kps = self.norm_last_kps[inliers]

        return model.params

    @staticmethod
    def normalize(k, pts):
        kinv = np.linalg.inv(k)
        add_ones = lambda x: np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        # 归一化公式
        """
        将相机坐标系下的角点坐标转换为在归一化坐标系下的表示，消除相机的投影畸变
        pts:角点坐标 
        .T:简单转置
        [0:2]:恢复坐标 
        """
        norm_pts = np.dot(kinv, add_ones(pts).T).T[:, 0:2]
        return norm_pts

    def triangulate(self):
        """已经知道像素坐标/相机内参/相机位姿，求世界坐标"""
        pose1 = np.linalg.inv(self.last_pose)  # 从世界坐标系变换到相机坐标系的位姿, 因此取逆
        pose2 = np.linalg.inv(self.now_pose)
        points4d = np.zeros((self.norm_last_kps.shape[0], 4))
        for i, (kp1, kp2) in enumerate(zip(self.norm_last_kps, self.norm_now_kps)):
            A = np.zeros((4, 4))
            # 角点和相机位姿带入方程
            A[0] = kp1[0] * pose1[2] - pose1[0]
            A[1] = kp1[1] * pose1[2] - pose1[1]
            A[2] = kp2[0] * pose2[2] - pose2[0]
            A[3] = kp2[1] * pose2[2] - pose2[1]
            _, _, vt = np.linalg.svd(A)  # 对 A 进行奇异值分解
            points4d[i] = vt[3]  # x=(u,v,1)

        points4d /= points4d[:, 3:]  # 归一化变换成齐次坐标 [x, y, z, 1]
        return points4d

    def draw_points(self):
        for kp1, kp2 in zip(self.now_kps, self.last_kps):
            u1, v1 = int(kp1[0]), int(kp1[1])
            u2, v2 = int(kp2[0]), int(kp2[1])
            cv2.circle(self.image, (u1, v1), color=(0, 0, 255), radius=3)
            cv2.line(self.image, (u1, v1), (u2, v2), color=(255, 0, 0))
        return None

    @staticmethod
    def check_points(points4d):
        # 判断3D点是否在两个摄像头前方
        good_points = points4d[:, 2] > 0
        return good_points

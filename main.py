import cv2
import numpy as np
import module.Frame as Frame
import module.optimizer as optimizer

if __name__ == "__main__":
    "主函数"
    W, H = 960, 540
    F = 270
    K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
    cap = cv2.VideoCapture("test.mp4")
    # 是否使用BA优化
    ba = True

    while cap.isOpened():
        ret, frame = cap.read()
        frame = Frame.Frame(frame, K)
        if ret:
            image, result = frame.process_frame()
            if result is not None and ba:
                opt = optimizer.BundleAdjustmentOptimer(result[0], result[1], result[2], K)
                frame.update(opt.update())
        else:
            break
        cv2.imshow("slam", image)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

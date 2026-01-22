# import cv2

# cap = cv2.VideoCapture(0)  # /dev/video0
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv2.CAP_PROP_FPS, 30)

# if cap.isOpened():
#     print("Camera opened successfully!")
# else:
#     print("Failed to open camera.")
# cap.release()
import cv2

def scan_cameras():
    print("--- 扫描所有可能的摄像头节点 ---")
    for i in range(10):  # 检查 0-9 号节点
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[SUCCESS] 节点 /dev/video{i}: 可打开且能读取图像")
            else:
                print(f"[WARNING] 节点 /dev/video{i}: 可打开但读取失败 (可能被占用或格式错误)")
            cap.release()
        else:
            pass # 节点不存在

scan_cameras()
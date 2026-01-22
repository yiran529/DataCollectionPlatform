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

def test_video0():
    """专门测试/dev/video0摄像头"""
    # 关键：用设备路径而非索引打开，指定CAP_V4L2后端
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    
    # 检查是否打开成功
    if not cap.isOpened():
        print("[ERROR] 无法打开/dev/video0")
        # 尝试打印错误原因（辅助排查）
        import traceback
        traceback.print_exc()
        return
    
    print("[SUCCESS] 成功打开/dev/video0")
    print("摄像头参数：")
    # 读取摄像头支持的参数
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    print(f"  分辨率：{width}x{height}")
    print(f"  帧率：{fps}")
    print(f"  编码格式：{fourcc}")
    
    # 尝试读取一帧
    ret, frame = cap.read()
    if ret:
        print("[SUCCESS] 成功读取/dev/video0的图像帧")
        # 保存一帧到本地，验证图像是否正常
        cv2.imwrite("video0_test.jpg", frame)
        print("  图像已保存为 video0_test.jpg")
    else:
        print("[WARNING] 能打开/dev/video0，但读取帧失败（可能格式不兼容）")
        # 尝试强制设置MJPG格式（v4l2显示video0用的是MJPG）
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret, frame = cap.read()
        if ret:
            print("[SUCCESS] 强制设置MJPG格式后，成功读取帧")
            cv2.imwrite("video0_test_mjpg.jpg", frame)
        else:
            print("[ERROR] 即使强制MJPG，仍无法读取帧")
    
    # 释放资源
    cap.release()

if __name__ == "__main__":
    test_video0()
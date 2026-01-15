import cv2
import os

def main():
    # 打开编号为4的摄像头
    cap = cv2.VideoCapture(6)
    if not cap.isOpened():
        print("无法打开摄像头 id: 4")
        return

    # 设置摄像头采集格式为 MJPG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # 设置分辨率和帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # 保存帧的目录
    left_dir = "stereo_frames_left"
    right_dir = "stereo_frames_right"
    if not os.path.exists(left_dir):
        os.makedirs(left_dir)
    if not os.path.exists(right_dir):
        os.makedirs(right_dir)

    frame_count = 0

    print("按'q'退出。")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.shape[1] < 3840 or frame.shape[0] < 1080:
            print("无法读取有效视频帧（可能帧为空或分辨率不足）")
            continue

        # 将3840x1080帧左右裁剪
        try:
            # 左相机图像
            left_img = frame[:, :1920, :]
            # 右相机图像
            right_img = frame[:, 1920:, :]

            # 检查左右图像是否都非空且尺寸正确
            if left_img.size == 0 or right_img.size == 0 or left_img.shape[1] != 1920 or right_img.shape[1] != 1920:
                print("左右图像尺寸异常，跳过该帧")
                continue

            # 保存左右图像
            left_filename = os.path.join(left_dir, f"left_{frame_count:06d}.png")
            right_filename = os.path.join(right_dir, f"right_{frame_count:06d}.png")
            success_left = cv2.imwrite(left_filename, left_img)
            success_right = cv2.imwrite(right_filename, right_img)
            if not success_left or not success_right:
                print(f"保存图像失败: {left_filename} 或 {right_filename}")
                continue
            frame_count += 1

        except Exception as e:
            print(f"裁剪或保存图像时出错: {e}")
            continue

        # 不保存为视频，只保存图像

        # 显示视频
        cv2.imshow('Stereo Camera 4', frame)

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
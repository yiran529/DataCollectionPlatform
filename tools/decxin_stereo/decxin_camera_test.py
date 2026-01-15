#!/usr/bin/env python3
"""
DECXIN 立体相机测试脚本
测试 3840x1080 @ 60fps 捕获能力
"""

import cv2
import time
import numpy as np
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading


def test_stereo_camera(device_id=4, width=3840, height=1080, target_fps=60, test_duration=5, show_preview=False):
    """
    测试立体相机的捕获能力
    
    Args:
        device_id: 视频设备ID (/dev/video4)
        width: 图像宽度
        height: 图像高度
        target_fps: 目标帧率
        test_duration: 测试时长（秒）
        show_preview: 是否显示实时预览
    """
    print(f"=" * 60)
    print(f"DECXIN 立体相机测试")
    print(f"目标: {width}x{height} @ {target_fps}fps")
    print(f"设备: /dev/video{device_id}")
    print(f"=" * 60)
    
    # 打开相机
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print("❌ 无法打开相机!")
        return False
    
    # 设置 MJPG 格式（必须先设置格式，再设置分辨率）
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 设置帧率
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    
    # 读取实际设置
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    print(f"\n实际设置:")
    print(f"  分辨率: {actual_width}x{actual_height}")
    print(f"  帧率: {actual_fps} fps")
    print(f"  格式: {fourcc_str}")
    
    # 预热（丢弃前几帧）
    print(f"\n预热中...")
    for _ in range(10):
        cap.read()
    
    # 开始帧率测试
    print(f"\n开始 {test_duration} 秒帧率测试...")
    frame_count = 0
    frame_times = []
    start_time = time.time()
    last_time = start_time
    
    while time.time() - start_time < test_duration:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            current_time = time.time()
            frame_times.append(current_time - last_time)
            last_time = current_time
            
            # 每秒打印一次状态
            if frame_count % target_fps == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                print(f"  已捕获 {frame_count} 帧, 当前帧率: {current_fps:.1f} fps")
            
            # 实时可视化
            if show_preview:
                # 缩小显示（原图太大）
                display = cv2.resize(frame, (1920, 540))
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(display, f"FPS: {current_fps:.1f} | Frame: {frame_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Stereo Camera Test", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户中断测试")
                    break
        else:
            print("  ⚠️  丢帧!")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 计算统计数据
    avg_fps = frame_count / total_time
    avg_frame_time = np.mean(frame_times) * 1000  # ms
    min_frame_time = np.min(frame_times) * 1000
    max_frame_time = np.max(frame_times) * 1000
    std_frame_time = np.std(frame_times) * 1000
    
    print(f"\n" + "=" * 60)
    print(f"测试结果:")
    print(f"  总帧数: {frame_count}")
    print(f"  测试时长: {total_time:.2f} 秒")
    print(f"  平均帧率: {avg_fps:.2f} fps")
    print(f"  帧间隔: {avg_frame_time:.2f} ms (平均)")
    print(f"  帧间隔: {min_frame_time:.2f} - {max_frame_time:.2f} ms (范围)")
    print(f"  帧间隔标准差: {std_frame_time:.2f} ms")
    print(f"=" * 60)
    
    # 判断是否达到目标
    if avg_fps >= target_fps * 0.95:  # 允许5%误差
        print(f"✅ 成功! 相机可以达到 {target_fps} fps")
        success = True
    else:
        print(f"❌ 未达到目标帧率 ({target_fps} fps)")
        success = False
    
    # 捕获并显示一帧用于验证
    ret, frame = cap.read()
    if ret:
        print(f"\n捕获的图像尺寸: {frame.shape}")
        
        # 分割左右图像
        mid = frame.shape[1] // 2
        left_img = frame[:, :mid]
        right_img = frame[:, mid:]
        print(f"  左图: {left_img.shape}")
        print(f"  右图: {right_img.shape}")
        
        # 保存示例图像
        cv2.imwrite("stereo_test_full.jpg", frame)
        cv2.imwrite("stereo_test_left.jpg", left_img)
        cv2.imwrite("stereo_test_right.jpg", right_img)
        print(f"\n已保存示例图像:")
        print(f"  - stereo_test_full.jpg (完整)")
        print(f"  - stereo_test_left.jpg (左)")
        print(f"  - stereo_test_right.jpg (右)")
    
    cap.release()
    if show_preview:
        cv2.destroyAllWindows()
    return success


def save_images_worker(save_queue, stop_event):
    """后台保存图像的工作线程"""
    while not stop_event.is_set() or not save_queue.empty():
        try:
            task = save_queue.get(timeout=0.1)
            if task is None:
                break
            filepath, image = task
            cv2.imwrite(filepath, image)
            save_queue.task_done()
        except:
            continue


def record_stereo_images(device_id=4, width=3840, height=1080, fps=60, 
                         duration=5, output_dir="record", use_jpg=True, show_preview=False):
    """
    录制立体相机图像序列（使用多线程保存以达到实时帧率）
    
    Args:
        device_id: 视频设备ID
        width: 图像宽度
        height: 图像高度
        fps: 帧率
        duration: 录制时长（秒）
        output_dir: 输出目录
        use_jpg: 使用JPEG格式（更快）还是PNG格式（无损）
        show_preview: 是否显示实时预览
    """
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(output_dir, f"session_{timestamp}")
    full_dir = os.path.join(base_dir, "full")
    left_dir = os.path.join(base_dir, "left")
    right_dir = os.path.join(base_dir, "right")
    
    # 创建目录
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    
    ext = ".jpg" if use_jpg else ".png"
    
    print(f"=" * 60)
    print(f"DECXIN 立体相机录制")
    print(f"分辨率: {width}x{height} @ {fps}fps")
    print(f"录制时长: {duration} 秒")
    print(f"图像格式: {'JPEG (高速)' if use_jpg else 'PNG (无损)'}")
    print(f"输出目录: {base_dir}")
    print(f"=" * 60)
    
    # 打开相机
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print("❌ 无法打开相机!")
        return False
    
    # 设置相机参数
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # 验证设置
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n相机设置: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    # 创建保存队列和工作线程
    save_queue = Queue(maxsize=fps * 10)  # 缓冲10秒的帧
    stop_event = threading.Event()
    
    # 启动多个保存线程
    num_workers = 8
    workers = []
    for _ in range(num_workers):
        t = threading.Thread(target=save_images_worker, args=(save_queue, stop_event))
        t.start()
        workers.append(t)
    
    print(f"启动 {num_workers} 个保存线程")
    
    # 预热
    print("预热中...")
    for _ in range(10):
        cap.read()
    
    # 开始录制
    print(f"\n开始录制 (按 Ctrl+C 提前停止)...")
    frame_count = 0
    start_time = time.time()
    timestamps = []
    
    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 丢帧!")
                continue
            
            # 记录时间戳
            frame_time = time.time()
            timestamps.append(frame_time - start_time)
            
            # 分割左右图像
            mid = frame.shape[1] // 2
            left_img = frame[:, :mid].copy()
            right_img = frame[:, mid:].copy()
            
            # 添加到保存队列
            frame_name = f"{frame_count:06d}{ext}"
            try:
                save_queue.put((os.path.join(full_dir, frame_name), frame.copy()), block=False)
                save_queue.put((os.path.join(left_dir, frame_name), left_img), block=False)
                save_queue.put((os.path.join(right_dir, frame_name), right_img), block=False)
            except:
                print(f"⚠️ 保存队列已满，跳过帧 {frame_count}")
            
            frame_count += 1
            
            # 实时可视化
            if show_preview:
                display = cv2.resize(frame, (1920, 540))
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                queue_size = save_queue.qsize()
                # 显示帧信息
                cv2.putText(display, f"REC | FPS: {current_fps:.1f} | Frame: {frame_count} | Queue: {queue_size}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # 显示左右分割线
                cv2.line(display, (960, 0), (960, 540), (0, 255, 0), 2)
                cv2.putText(display, "LEFT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display, "RIGHT", (970, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("Recording Preview", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户中断录制")
                    break
            
            # 每秒打印进度
            if frame_count % fps == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                queue_size = save_queue.qsize()
                print(f"  已录制 {frame_count} 帧 ({elapsed:.1f}s), 帧率: {current_fps:.1f} fps, 队列: {queue_size}")
    
    except KeyboardInterrupt:
        print("\n用户中断录制")
    
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        # 等待保存完成
        print("\n等待图像保存完成...")
        stop_event.set()
        for t in workers:
            t.join()
    
    # 保存时间戳
    timestamp_file = os.path.join(base_dir, "timestamps.txt")
    with open(timestamp_file, 'w') as f:
        f.write("# frame_index timestamp_seconds\n")
        for i, ts in enumerate(timestamps):
            f.write(f"{i} {ts:.6f}\n")
    
    # 统计
    total_time = timestamps[-1] if timestamps else 0
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print(f"\n" + "=" * 60)
    print(f"录制完成!")
    print(f"  总帧数: {frame_count}")
    print(f"  录制时长: {total_time:.2f} 秒")
    print(f"  平均帧率: {avg_fps:.2f} fps")
    print(f"\n保存位置:")
    print(f"  完整图像: {full_dir}/ ({frame_count} 张)")
    print(f"  左相机:   {left_dir}/ ({frame_count} 张)")
    print(f"  右相机:   {right_dir}/ ({frame_count} 张)")
    print(f"  时间戳:   {timestamp_file}")
    print(f"=" * 60)
    
    return True


def show_live_preview(device_id=4, width=3840, height=1080, fps=60):
    """显示实时预览（按 'q' 退出）"""
    print("正在打开实时预览... (按 'q' 退出)")
    
    cap = cv2.VideoCapture(device_id)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # 缩小显示（原图太大）
        display = cv2.resize(frame, (1920, 540))
        
        # 添加帧率信息
        cv2.putText(display, f"FPS: {current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Stereo Camera Preview", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DECXIN 立体相机测试")
    parser.add_argument("--device", "-d", type=int, default=6, help="视频设备ID")
    parser.add_argument("--width", "-W", type=int, default=3840, help="图像宽度")
    parser.add_argument("--height", "-H", type=int, default=1080, help="图像高度")
    parser.add_argument("--fps", "-f", type=int, default=60, help="目标帧率")
    parser.add_argument("--duration", "-t", type=int, default=5, help="测试/录制时长(秒)")
    parser.add_argument("--preview", "-p", action="store_true", help="显示实时预览")
    parser.add_argument("--record", "-r", action="store_true", help="录制图像序列")
    parser.add_argument("--output", "-o", type=str, default="record", help="录制输出目录")
    parser.add_argument("--show", "-s", action="store_true", help="测试/录制时显示可视化窗口")
    
    args = parser.parse_args()
    
    if args.preview:
        show_live_preview(args.device, args.width, args.height, args.fps)
    elif args.record:
        record_stereo_images(args.device, args.width, args.height, args.fps, 
                            args.duration, args.output, show_preview=args.show)
    else:
        test_stereo_camera(args.device, args.width, args.height, args.fps, args.duration, 
                          show_preview=args.show)

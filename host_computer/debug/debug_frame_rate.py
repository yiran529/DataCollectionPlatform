#!/usr/bin/env python3
"""
帧率对标诊断：对比 v4l2-ctl 设置 vs OpenCV 实际
找出 USB 带宽限制导致的自适应调整

在 Jetson 上运行：
  python3 debug_frame_rate.py --device 0 --test-fps 30 60 20 15 10 5 1
"""

import sys
import os
import cv2
import time
import argparse
import subprocess

def get_v4l2_fps(device_id):
    """用 v4l2-ctl 读取当前帧率设置"""
    try:
        result = subprocess.run(
            ['v4l2-ctl', '-d', f'/dev/video{device_id}', '--get-fmt-video'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=3
        )
        # 解析输出找到帧率信息
        for line in result.stdout.split('\n'):
            if 'Interval:' in line or 'fps' in line:
                return line.strip()
    except Exception as e:
        print(f"⚠️ v4l2-ctl 读取失败: {e}")
    return "未知"

def test_fps(device_id, width, height, target_fps):
    """测试单个帧率配置"""
    # 用 v4l2-ctl 设置帧率
    subprocess.run(
        ['v4l2-ctl', '-d', f'/dev/video{device_id}', 
         '--set-fmt-video', f'width={width},height={height},pixelformat=MJPG'],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    subprocess.run(
        ['v4l2-ctl', '-d', f'/dev/video{device_id}', '-p', str(int(target_fps))],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    # 打开 OpenCV
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    
    time.sleep(0.1)
    
    # v4l2-ctl 重新配置
    subprocess.run(
        ['v4l2-ctl', '-d', f'/dev/video{device_id}', 
         '--set-fmt-video', f'width={width},height={height},pixelformat=MJPG'],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    subprocess.run(
        ['v4l2-ctl', '-d', f'/dev/video{device_id}', '-p', str(int(target_fps))],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    time.sleep(0.1)
    
    # OpenCV 设置帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    
    time.sleep(0.1)
    
    # 读取实际参数
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 用 v4l2-ctl 读取驱动侧的帧率
    v4l2_info = get_v4l2_fps(device_id)
    
    # 测试帧率（实际读取几帧看是否稳定）
    frame_times = []
    last_time = time.time()
    frame_count = 0
    start_time = time.time()
    
    while time.time() - start_time < 1.0 and frame_count < 10:  # 采样1秒或最多10帧
        ret, frame = cap.read()
        if ret:
            current_time = time.time()
            if frame_count > 0:
                frame_times.append(current_time - last_time)
            last_time = current_time
            frame_count += 1
    
    cap.release()
    
    # 计算实际采集帧率
    if len(frame_times) > 2:
        import statistics
        avg_interval = statistics.mean(frame_times)
        measured_fps = 1.0 / avg_interval if avg_interval > 0 else 0
        fps_std = statistics.stdev(frame_times) if len(frame_times) > 1 else 0
    else:
        measured_fps = 0
        fps_std = 0
    
    return {
        'target': target_fps,
        'opencv_get': actual_fps,
        'measured': measured_fps,
        'frame_count': frame_count,
        'fps_std': fps_std,
        'v4l2_info': v4l2_info
    }

def main():
    parser = argparse.ArgumentParser(description="帧率对标诊断")
    parser.add_argument("--device", "-d", type=int, default=0, help="设备号")
    parser.add_argument("--width", "-w", type=int, default=3840, help="分辨率宽")
    parser.add_argument("--height", "-H", type=int, default=1080, help="分辨率高")
    parser.add_argument("--test-fps", type=int, nargs='+', 
                       default=[60, 30, 20, 15, 10, 5, 1],
                       help="要测试的帧率列表")
    args = parser.parse_args()
    
    device_path = f"/dev/video{args.device}"
    if not os.path.exists(device_path):
        print(f"❌ 设备不存在: {device_path}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"帧率对标诊断 - /dev/video{args.device}")
    print("=" * 80)
    print(f"分辨率: {args.width}x{args.height}")
    print(f"测试帧率: {args.test_fps}\n")
    
    results = []
    for target_fps in args.test_fps:
        print(f"测试 {target_fps} fps...", end='', flush=True)
        result = test_fps(args.device, args.width, args.height, target_fps)
        
        if result is None:
            print(" ❌ 无法打开设备")
            continue
        
        results.append(result)
        print(f" ✓")
        time.sleep(0.5)  # 两次测试间隔
    
    # 输出结果表
    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)
    print(f"{'请求':>8} | {'OpenCV':>10} | {'实测':>10} | {'帧数':>4} | {'稳定性':>8} | 说明")
    print("-" * 80)
    
    for r in results:
        target = r['target']
        opencv = r['opencv_get']
        measured = r['measured']
        frame_count = r['frame_count']
        fps_std = r['fps_std']
        
        # 判断状态
        if opencv == target and measured >= target * 0.9:
            status = "✅ 支持"
        elif opencv < target:
            status = "⚠️ 驱动降低"
        else:
            status = "❓ 异常"
        
        print(f"{target:>6}fps | {opencv:>8.1f}fps | {measured:>8.1f}fps | {frame_count:>4} | "
              f"{fps_std:>6.3f}ms | {status}")
    
    # 带宽计算
    print("\n" + "=" * 80)
    print("💡 带宽分析")
    print("=" * 80)
    
    print(f"""
MJPEG 压缩比估算（3840x1080 图像内容相关）：
  - 复杂场景（高压缩率）: ~1.5-2 Mbps/fps
  - 一般场景（中压缩率）: ~2.5-4 Mbps/fps  
  - 简单场景（低压缩率）: ~4-6 Mbps/fps

当前配置估算（假设中等压缩）：
  - 3840x1080 @ 30fps: ~90-120 Mbps
  - 3840x1080 @ 1fps:  ~3-4 Mbps
  
USB 2.0 可用带宽: ~350 Mbps

如果实测帧率远低于请求帧率，说明：
  ✓ 硬件支持该分辨率和帧率
  ✓ 但 USB 带宽可能在双摄像头场景下不足
  
解决方案：
  1️⃣ 降低分辨率（3840x1080 → 1920x1080 或 1280x720）
  2️⃣ 降低帧率（30fps → 15fps 或 5fps）
  3️⃣ 禁用其他摄像头单独测试
  4️⃣ 检查另一个摄像头的配置是否过高
""")

if __name__ == "__main__":
    main()

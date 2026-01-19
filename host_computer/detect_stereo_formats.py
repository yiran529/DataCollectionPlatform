#!/usr/bin/env python3
"""
检查 Stereo 摄像头支持的格式和分辨率
"""

import cv2
import subprocess
import sys

print("=" * 70)
print("Stereo 摄像头格式检测")
print("=" * 70)

device = 0
print(f"\n检查 /dev/video{device}...")

# 方法1: 使用 v4l2-ctl 获取详细信息
print("\n[方法1] v4l2-ctl 详细信息:")
print("-" * 70)

try:
    result = subprocess.run(
        ['v4l2-ctl', '-d', f'/dev/video{device}', '--list-formats-ext'],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("❌ 命令失败")
        if result.stderr:
            print(result.stderr)
except Exception as e:
    print(f"❌ 错误: {e}")

# 方法2: 使用 OpenCV 尝试不同的分辨率
print("\n[方法2] OpenCV 测试不同分辨率:")
print("-" * 70)

test_configs = [
    (3840, 1080, 1, "原始配置（3840x1080 @ 1fps）"),
    (1920, 540, 5, "降级配置1（1920x540 @ 5fps）"),
    (1920, 1080, 5, "降级配置2（1920x1080 @ 5fps）"),
    (1280, 720, 10, "降级配置3（1280x720 @ 10fps）"),
    (640, 480, 30, "最低配置（640x480 @ 30fps）"),
]

successful_configs = []

for width, height, fps, desc in test_configs:
    print(f"\n试试 {desc}...", end='', flush=True)
    
    cap = cv2.VideoCapture(device)
    
    if not cap.isOpened():
        print(" ❌ 无法打开设备")
        continue
    
    # 设置参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减小缓冲
    
    # 获取实际参数
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 尝试读取帧（超时3秒）
    import time
    start = time.time()
    ret = None
    frame = None
    timeout = 3.0
    
    while time.time() - start < timeout:
        ret, frame = cap.read()
        if ret:
            break
    
    cap.release()
    
    if ret and frame is not None:
        actual_shape = frame.shape
        print(f" ✓")
        print(f"    请求: {width}x{height} @ {fps}fps")
        print(f"    实际: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")
        print(f"    帧大小: {actual_shape}")
        successful_configs.append({
            'width': actual_w,
            'height': actual_h,
            'fps': actual_fps,
            'desc': desc
        })
    else:
        print(f" ✗（无法读取帧，可能超时）")
        print(f"    实际参数: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")

print("\n" + "=" * 70)
print("总结")
print("=" * 70)

if successful_configs:
    print(f"\n✓ 可用配置（{len(successful_configs)} 个）:")
    for i, cfg in enumerate(successful_configs, 1):
        print(f"{i}. {cfg['desc']}")
        print(f"   {cfg['width']}x{cfg['height']} @ {cfg['fps']:.1f}fps")
    
    print("\n📝 推荐配置：")
    best = successful_configs[0]  # 第一个是最高分辨率
    print(f"\nstereo:")
    print(f"  device: {device}")
    print(f"  width: {best['width']}")
    print(f"  height: {best['height']}")
    print(f"  fps: {int(best['fps'])}")
else:
    print("\n❌ 没有可用配置")
    print("可能原因：")
    print("  1. 摄像头本身有问题")
    print("  2. USB 供电不足")
    print("  3. 摄像头与 Mono 摄像头冲突")

print("\n" + "=" * 70)

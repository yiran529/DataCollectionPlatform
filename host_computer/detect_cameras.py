#!/usr/bin/env python3
"""
检测 Jetson 上的摄像头设备
找到正确的设备号以配置 config.yaml
"""

import os
import subprocess
import cv2

print("=" * 70)
print("Jetson Xavier 摄像头检测工具")
print("=" * 70)

# 方法1: 使用 v4l2-ctl 列出所有设备
print("\n[方法1] 使用 v4l2-ctl 检测设备...")
print("-" * 70)

try:
    result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("v4l2-ctl 命令失败，请先安装: sudo apt install v4l-utils")
except FileNotFoundError:
    print("❌ v4l2-ctl 未安装")
except Exception as e:
    print(f"❌ 错误: {e}")

# 方法2: 检查 /dev/video* 文件
print("\n[方法2] 检查 /dev/video* 设备...")
print("-" * 70)

video_devices = []
for i in range(20):  # 检查 /dev/video0 到 /dev/video19
    dev_path = f'/dev/video{i}'
    if os.path.exists(dev_path):
        video_devices.append(i)
        print(f"  ✓ {dev_path} 存在")

if not video_devices:
    print("  ❌ 未找到任何 /dev/video* 设备")
    print("  请检查摄像头硬件连接")

# 方法3: 尝试打开每个设备测试是否可用
print("\n[方法3] 测试摄像头可用性...")
print("-" * 70)

available_cameras = []

for device_id in video_devices:
    dev_path = f'/dev/video{device_id}'
    
    # 使用 OpenCV 尝试打开（标准方式）
    print(f"\n  测试 {dev_path}:")
    
    cap = cv2.VideoCapture(device_id)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"    ✓ 可以打开")
        print(f"      分辨率: {width}x{height}")
        print(f"      FPS: {fps:.1f}")
        
        # 尝试读取一帧
        ret, frame = cap.read()
        if ret:
            print(f"    ✓ 可以读取帧 (shape: {frame.shape})")
            available_cameras.append({
                'id': device_id,
                'path': dev_path,
                'width': width,
                'height': height,
                'fps': fps
            })
        else:
            print(f"    ⚠ 可以打开但无法读取帧")
        
        cap.release()
    else:
        print(f"    ✗ 无法打开")

# 方法4: 尝试 GStreamer 管道（Jetson 硬件加速）
print("\n[方法4] 测试 GStreamer 硬件加速...")
print("-" * 70)

for device_id in video_devices:
    dev_path = f'/dev/video{device_id}'
    
    print(f"\n  测试 {dev_path} (GStreamer):")
    
    # 尝试 GStreamer 管道
    gst_pipeline = (
        f"v4l2src device={dev_path} ! "
        f"image/jpeg,width=1920,height=1080,framerate=30/1 ! "
        f"jpegdec ! videoconvert ! appsink"
    )
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"    ✓ GStreamer 可以打开")
        ret, frame = cap.read()
        if ret:
            print(f"    ✓ 可以读取帧 (shape: {frame.shape})")
        cap.release()
    else:
        print(f"    ✗ GStreamer 无法打开 (可能摄像头不支持该分辨率/格式)")

# 输出总结
print("\n" + "=" * 70)
print("总结")
print("=" * 70)

if available_cameras:
    print(f"\n✓ 检测到 {len(available_cameras)} 个可用摄像头:")
    for cam in available_cameras:
        print(f"  - 设备 {cam['id']}: {cam['width']}x{cam['height']} @ {cam['fps']:.1f}fps")
    
    print("\n📝 更新 config.yaml：")
    print()
    print("# 根据检测结果修改以下配置:")
    print("left_hand:")
    print("  mono:")
    if len(available_cameras) >= 1:
        print(f"    device: {available_cameras[0]['id']}  # 改为检测到的设备号")
    else:
        print("    device: 0  # 改为检测到的设备号")
    print("    width: 1600")
    print("    height: 1200")
    print("    fps: 30")
    print("  stereo:")
    if len(available_cameras) >= 2:
        print(f"    device: {available_cameras[1]['id']}  # 改为检测到的设备号")
    else:
        print("    device: 2  # 改为检测到的设备号")
    print("    width: 3840")
    print("    height: 1080")
    print("    fps: 30")
else:
    print("\n❌ 未检测到任何可用摄像头")
    print("\n故障排查:")
    print("  1. 检查 USB 摄像头是否正确连接")
    print("  2. 检查电源是否充足（某些摄像头需要更多电流）")
    print("  3. 运行: lsusb | grep -i camera")
    print("  4. 运行: dmesg | tail -20  # 查看系统日志")
    print("  5. 尝试重新插拔 USB 摄像头")

print("\n" + "=" * 70)

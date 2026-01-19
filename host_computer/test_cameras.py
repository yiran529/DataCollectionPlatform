#!/usr/bin/env python3
"""
测试摄像头采集（不使用 GStreamer 硬件加速）
"""

import sys
import os

# 确保不使用 GStreamer
os.environ['USE_GSTREAMER'] = '0'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import yaml
from data_coll.sync_data_collector import CameraReader

print("=" * 70)
print("摄像头采集测试（标准 V4L2 模式）")
print("=" * 70)

# 加载配置
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

right_hand_cfg = config['right_hand']
stereo_cfg = right_hand_cfg['stereo']
mono_cfg = right_hand_cfg['mono']

print(f"\n[1] 测试 Mono 摄像头 (设备 {mono_cfg['device']})...")
print("-" * 70)

mono = CameraReader(
    device_id=mono_cfg['device'],
    width=mono_cfg['width'],
    height=mono_cfg['height'],
    fps=mono_cfg['fps'],
    name="MONO"
)

if mono.open():
    print("✓ Mono 摄像头打开成功")
    mono.start()
    
    # 读取几帧
    import time
    time.sleep(0.5)
    
    buffer = mono.buffer
    print(f"✓ 已采集 {len(buffer)} 帧")
    
    if buffer:
        frame = buffer[-1]
        print(f"✓ 最后一帧: {frame.frame.shape}, ts={frame.timestamp:.3f}")
    
    mono.stop()
    print("✓ Mono 摄像头关闭")
else:
    print("❌ Mono 摄像头打开失败")

print(f"\n[2] 测试 Stereo 摄像头 (设备 {stereo_cfg['device']})...")
print("-" * 70)

stereo = CameraReader(
    device_id=stereo_cfg['device'],
    width=stereo_cfg['width'],
    height=stereo_cfg['height'],
    fps=stereo_cfg['fps'],
    name="STEREO"
)

if stereo.open():
    print("✓ Stereo 摄像头打开成功")
    stereo.start()
    
    # 读取几帧
    import time
    time.sleep(0.5)
    
    buffer = stereo.buffer
    print(f"✓ 已采集 {len(buffer)} 帧")
    
    if buffer:
        frame = buffer[-1]
        print(f"✓ 最后一帧: {frame.frame.shape}, ts={frame.timestamp:.3f}")
    
    stereo.stop()
    print("✓ Stereo 摄像头关闭")
else:
    print("❌ Stereo 摄像头打开失败")

print("\n" + "=" * 70)
print("✓ 摄像头采集测试完成")
print("=" * 70)

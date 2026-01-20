#!/usr/bin/env python3
"""
双摄像头带宽诊断
测试两个摄像头同时工作时的参数变化

在 Jetson 上运行：
  python3 debug_dual_camera_bandwidth.py
"""

import sys
import os
import cv2
import time
import subprocess

def configure_and_test(device_id, width, height, target_fps, name):
    """配置并测试单个摄像头"""
    
    # v4l2-ctl 配置
    subprocess.run(
        ['v4l2-ctl', '-d', f'/dev/video{device_id}', 
         '--set-fmt-video', f'width={width},height={height},pixelformat=MJPG'],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    subprocess.run(
        ['v4l2-ctl', '-d', f'/dev/video{device_id}', '-p', str(int(target_fps))],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    # OpenCV 打开
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"  ❌ 无法打开 {name}")
        return None
    
    time.sleep(0.1)
    
    # OpenCV 重新配置
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    
    time.sleep(0.1)
    
    # 读取实际参数
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 预估带宽（假设 MJPEG 平均压缩率 50%）
    pixel_count = actual_w * actual_h
    bytes_per_frame = int(pixel_count * 1.5 * 0.5)  # YUV420 + MJPEG 压缩
    bandwidth_mbps = (bytes_per_frame * actual_fps * 8) / 1_000_000
    
    result = {
        'name': name,
        'device': device_id,
        'requested': f"{width}x{height} @ {target_fps}fps",
        'actual': f"{actual_w}x{actual_h} @ {actual_fps:.1f}fps",
        'bandwidth_mbps': bandwidth_mbps,
        'cap': cap
    }
    
    return result

def main():
    print("=" * 80)
    print("双摄像头带宽诊断")
    print("=" * 80)
    
    # 配置
    stereo_cfg = {
        'device': 0,
        'width': 3840,
        'height': 1080,
        'fps': 30,
        'name': 'STEREO'
    }
    
    mono_cfg = {
        'device': 1,
        'width': 1600,
        'height': 1300,
        'fps': 30,
        'name': 'MONO'
    }
    
    print("\n1️⃣ 单独测试 Stereo:")
    print("-" * 80)
    stereo_solo = configure_and_test(
        stereo_cfg['device'], stereo_cfg['width'], stereo_cfg['height'], 
        stereo_cfg['fps'], stereo_cfg['name']
    )
    if stereo_solo:
        print(f"  请求: {stereo_solo['requested']}")
        print(f"  实际: {stereo_solo['actual']}")
        print(f"  带宽: {stereo_solo['bandwidth_mbps']:.1f} Mbps")
        stereo_solo['cap'].release()
    
    time.sleep(1)
    
    print("\n2️⃣ 单独测试 Mono:")
    print("-" * 80)
    mono_solo = configure_and_test(
        mono_cfg['device'], mono_cfg['width'], mono_cfg['height'], 
        mono_cfg['fps'], mono_cfg['name']
    )
    if mono_solo:
        print(f"  请求: {mono_solo['requested']}")
        print(f"  实际: {mono_solo['actual']}")
        print(f"  带宽: {mono_solo['bandwidth_mbps']:.1f} Mbps")
        mono_solo['cap'].release()
    
    time.sleep(1)
    
    print("\n3️⃣ 同时打开两个摄像头:")
    print("-" * 80)
    stereo = configure_and_test(
        stereo_cfg['device'], stereo_cfg['width'], stereo_cfg['height'], 
        stereo_cfg['fps'], stereo_cfg['name']
    )
    
    time.sleep(0.5)
    
    mono = configure_and_test(
        mono_cfg['device'], mono_cfg['width'], mono_cfg['height'], 
        mono_cfg['fps'], mono_cfg['name']
    )
    
    if stereo and mono:
        print(f"\n  {stereo['name']}:")
        print(f"    请求: {stereo['requested']}")
        print(f"    实际: {stereo['actual']}")
        print(f"    带宽: {stereo['bandwidth_mbps']:.1f} Mbps")
        
        print(f"\n  {mono['name']}:")
        print(f"    请求: {mono['requested']}")
        print(f"    实际: {mono['actual']}")
        print(f"    带宽: {mono['bandwidth_mbps']:.1f} Mbps")
        
        total_bandwidth = stereo['bandwidth_mbps'] + mono['bandwidth_mbps']
        print(f"\n  总带宽: {total_bandwidth:.1f} Mbps")
        
        stereo['cap'].release()
        mono['cap'].release()
    
    # 分析
    print("\n" + "=" * 80)
    print("📊 分析结果")
    print("=" * 80)
    
    if stereo_solo and mono_solo and stereo and mono:
        print(f"""
单个摄像头状态：
  Stereo:  {stereo_solo['actual']}  (带宽: {stereo_solo['bandwidth_mbps']:.1f} Mbps)
  Mono:    {mono_solo['actual']}  (带宽: {mono_solo['bandwidth_mbps']:.1f} Mbps)

双摄像头状态：
  Stereo:  {stereo['actual']}  (带宽: {stereo['bandwidth_mbps']:.1f} Mbps)
  Mono:    {mono['actual']}  (带宽: {mono['bandwidth_mbps']:.1f} Mbps)
  总和:    {stereo['bandwidth_mbps'] + mono['bandwidth_mbps']:.1f} Mbps / 350 Mbps

""")
        
        # 判断问题
        if "1.0fps" in stereo_solo['actual']:
            print("⚠️ 问题：Stereo 摄像头单独测试就被限制在 1fps")
            print("   → 这是硬件驱动的固定行为（该摄像头可能默认就是 1fps）")
            print("   → 或 USB 链接本身有问题")
            print("")
            print("建议排查：")
            print("  1. 检查是否有其他程序占用摄像头")
            print("  2. 尝试重启 Jetson")
            print("  3. 检查 USB 线缆和接口是否松动")
            print("  4. 在另一个 USB 端口上测试")
        
        if stereo and "1.0fps" in stereo['actual'] and mono and "1.0fps" in mono['actual']:
            print("⚠️ 两个摄像头都被限制在 1fps")
            print("   → 很可能是 USB 设备树或驱动默认配置")
            print("")
            print("可尝试的修复：")
            print("  1. 查看 /sys/module/usbcore/parameters/usbfs_memory_mb")
            print("  2. 增加 USB 缓冲区：echo 0 > /sys/module/usbcore/parameters/usbfs_memory_mb")
            print("  3. 检查设备树配置（Jetson 特定）")

if __name__ == "__main__":
    main()

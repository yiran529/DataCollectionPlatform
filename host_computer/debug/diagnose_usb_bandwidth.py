#!/usr/bin/env python3
"""
USB 带宽诊断和摄像头格式检测
在 Jetson 上运行此脚本以获取真实的硬件能力
"""

import subprocess
import os
import sys

def run_cmd(cmd):
    """运行命令并返回输出"""
    try:
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=5)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return f"❌ 命令超时: {cmd}"
    except Exception as e:
        return f"❌ 错误: {e}"

def check_video_device(device_id):
    """检查单个视频设备"""
    device_path = f"/dev/video{device_id}"
    
    if not os.path.exists(device_path):
        return None
    
    print(f"\n{'='*70}")
    print(f"📹 设备: {device_path}")
    print('='*70)
    
    # 获取设备名称
    output = run_cmd(f"v4l2-ctl -d {device_path} --info")
    print(output)
    
    # 获取支持的格式和分辨率
    print("\n📋 支持的格式和分辨率:")
    print("-" * 70)
    output = run_cmd(f"v4l2-ctl -d {device_path} --list-formats-ext")
    
    # 解析输出，只显示前30行（避免过多信息）
    lines = output.split('\n')[:30]
    for line in lines:
        if line.strip():
            print(line)
    
    # 获取当前配置
    print("\n⚙️ 当前配置:")
    print("-" * 70)
    output = run_cmd(f"v4l2-ctl -d {device_path} --get-fmt-video")
    print(output)

def main():
    print("=" * 70)
    print("Jetson Xavier USB 摄像头诊断工具")
    print("=" * 70)
    
    # 检查是否为 Jetson
    if not os.path.exists('/etc/nv_tegra_release'):
        print("\n⚠️ 警告：此脚本应在 Jetson 设备上运行")
        print("未检测到 /etc/nv_tegra_release")
    else:
        with open('/etc/nv_tegra_release', 'r') as f:
            print(f"\n✓ Jetson 信息:\n{f.read()}")
    
    # 检查 USB 设备
    print("\n" + "=" * 70)
    print("🔌 USB 设备信息")
    print("=" * 70)
    output = run_cmd("lsusb")
    print(output)
    
    # 检查 USB 总线速度
    print("\n" + "=" * 70)
    print("⚡ USB 总线速度")
    print("=" * 70)
    output = run_cmd("cat /sys/bus/usb/devices/*/speed 2>/dev/null | sort | uniq -c")
    if output.strip():
        print(output)
    else:
        print("⚠️ 无法读取 USB 总线速度")
    
    # 扫描视频设备
    print("\n" + "=" * 70)
    print("📹 视频设备扫描")
    print("=" * 70)
    
    found_devices = []
    for i in range(0, 10):
        if os.path.exists(f"/dev/video{i}"):
            found_devices.append(i)
    
    if not found_devices:
        print("❌ 未找到任何视频设备")
        sys.exit(1)
    
    print(f"✓ 找到视频设备: {found_devices}")
    
    for device_id in found_devices:
        check_video_device(device_id)
    
    # 带宽计算
    print("\n" + "=" * 70)
    print("📊 USB 带宽估算")
    print("=" * 70)
    print("""
USB 2.0 High-Speed (典型)：
  - 理论最大带宽: 480 Mbps
  - 实际可用: ~350-400 Mbps
  
常见格式的带宽需求：
  - MJPEG 3840x1080 @ 30fps: ~150-200 Mbps（压缩率取决于内容）
  - MJPEG 1600x1300 @ 30fps: ~80-120 Mbps
  - YUYV 1920x1080 @ 30fps: ~746 Mbps（未压缩，不可行）
  
⚠️ 如果两个摄像头总带宽超过 350 Mbps，驱动会自动降低参数以避免丢帧。

✓ 解决方案：
  1. 降低分辨率或帧率
  2. 使用更高压缩率的 MJPEG
  3. 使用 USB 3.0 设备（如果 Jetson 支持）
  4. 尝试禁用不必要的摄像头功能
""")

    # 实际观测建议
    print("\n" + "=" * 70)
    print("💡 诊断步骤（在 Jetson 上执行）")
    print("=" * 70)
    print("""
1️⃣ 单独测试每个摄像头（禁用另一个）：
   python3 -c "
   import cv2
   cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
   cap.set(cv2.CAP_PROP_FPS, 30)
   
   w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   fps = cap.get(cv2.CAP_PROP_FPS)
   
   print(f'实际分辨率: {w}x{h} @ {fps}fps')
   cap.release()
   "

2️⃣ 检查驱动自动调整：
   v4l2-ctl -d /dev/video0 --get-fmt-video
   
3️⃣ 强制设置并观察：
   v4l2-ctl -d /dev/video0 --set-fmt-video=width=3840,height=1080,pixelformat=MJPG
   v4l2-ctl -d /dev/video0 --get-fmt-video
   
4️⃣ 测试实际数据流（查看丢帧）：
   timeout 5 ffplay -f v4l2 /dev/video0 -probesize 32
""")

if __name__ == "__main__":
    main()

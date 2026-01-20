#!/usr/bin/env python3
"""
摄像头参数调试脚本
用于诊断参数不匹配问题，不改动真正的采集代码

在 Jetson 上运行：
  python3 debug_camera_params.py --hand right
"""

import sys
import os
import cv2
import time
import yaml
import argparse
import subprocess

def test_camera(device_id, width, height, fps, name):
    """测试单个摄像头参数"""
    print(f"\n{'='*70}")
    print(f"测试 {name} - /dev/video{device_id}")
    print('='*70)
    
    # 第一步：用 v4l2-ctl 检查驱动支持
    print(f"\n1️⃣ v4l2-ctl 驱动检查:")
    try:
        result = subprocess.run(
            ['v4l2-ctl', '-d', f'/dev/video{device_id}', '--list-formats-ext'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=3
        )
        output = result.stdout.split('\n')
        # 只显示前 20 行
        for line in output[:20]:
            if line.strip():
                print(f"  {line}")
    except Exception as e:
        print(f"  ⚠️ v4l2-ctl 查询失败: {e}")
    
    # 第二步：用 v4l2-ctl 预先配置
    print(f"\n2️⃣ v4l2-ctl 预配置:")
    try:
        subprocess.run(
            ['v4l2-ctl', '-d', f'/dev/video{device_id}', 
             '--set-fmt-video', f'width={width},height={height},pixelformat=MJPG'],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        subprocess.run(
            ['v4l2-ctl', '-d', f'/dev/video{device_id}', '-p', str(int(fps))],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(f"  ✓ 已配置: {width}x{height} @ {fps}fps MJPG")
    except Exception as e:
        print(f"  ⚠️ v4l2-ctl 配置失败: {e}")
    
    # 第三步：用 OpenCV 打开
    print(f"\n3️⃣ OpenCV 打开设备:")
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"  ❌ 无法打开 /dev/video{device_id}")
        return
    
    print(f"  ✓ 设备已打开")
    time.sleep(0.2)
    
    # 第四步：用 v4l2-ctl 重新配置（因为 OpenCV 打开时会重置）
    print(f"\n4️⃣ v4l2-ctl 重新配置:")
    try:
        subprocess.run(
            ['v4l2-ctl', '-d', f'/dev/video{device_id}', 
             '--set-fmt-video', f'width={width},height={height},pixelformat=MJPG'],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        subprocess.run(
            ['v4l2-ctl', '-d', f'/dev/video{device_id}', '-p', str(int(fps))],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(f"  ✓ 已重新配置")
    except Exception as e:
        print(f"  ⚠️ v4l2-ctl 重新配置失败: {e}")
    
    time.sleep(0.2)
    
    # 第五步：OpenCV 设置参数
    print(f"\n5️⃣ OpenCV 设置参数:")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    print(f"  set() 已调用")
    
    time.sleep(0.1)
    
    # 第六步：读取实际参数
    print(f"\n6️⃣ 读取实际参数:")
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  请求:  {width:4}x{height:4} @ {fps:2}fps")
    print(f"  实际:  {actual_w:4}x{actual_h:4} @ {actual_fps:5.1f}fps")
    
    if actual_w == width and actual_h == height and actual_fps == fps:
        print(f"  ✅ 参数完全匹配！")
    else:
        print(f"  ❌ 参数不匹配！")
        if actual_fps != fps:
            print(f"     💡 帧率不符 - 驱动可能不支持或 USB 带宽不足")
        if actual_w != width or actual_h != height:
            print(f"     💡 分辨率不符 - 驱动自动降低以节省带宽")
    
    # 第七步：尝试读取帧
    print(f"\n7️⃣ 尝试读取帧:")
    try:
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                print(f"  ✓ 第 {i+1} 帧读取成功，形状: {frame.shape}")
                if i == 0:
                    break
            else:
                print(f"  ✗ 第 {i+1} 帧读取失败")
    except Exception as e:
        print(f"  ✗ 读取异常: {e}")
    
    cap.release()
    print(f"\n设备已关闭")

def main():
    parser = argparse.ArgumentParser(description="摄像头参数调试")
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                       help="配置文件路径")
    parser.add_argument("--hand", "-H", type=str, choices=['left', 'right'],
                       default='right', help="选择左手或右手")
    parser.add_argument("--stereo-only", "-s", action="store_true",
                       help="仅测试 Stereo 摄像头")
    parser.add_argument("--mono-only", "-m", action="store_true",
                       help="仅测试 Mono 摄像头")
    args = parser.parse_args()
    
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        hand_config = config.get(f'{args.hand}_hand', {})
        if not hand_config:
            print(f"❌ 配置中没有找到 {args.hand}_hand")
            sys.exit(1)
        
        print("=" * 70)
        print(f"摄像头参数调试 - {args.hand.upper()} 手")
        print("=" * 70)
        print(f"配置文件: {config_path}\n")
        
        # 测试 Stereo
        if not args.mono_only:
            stereo_cfg = hand_config.get('stereo', {})
            test_camera(
                stereo_cfg.get('device', 0),
                stereo_cfg.get('width', 3840),
                stereo_cfg.get('height', 1080),
                stereo_cfg.get('fps', 30),
                f"{args.hand.upper()}_STEREO"
            )
        
        # 测试 Mono
        if not args.stereo_only:
            mono_cfg = hand_config.get('mono', {})
            test_camera(
                mono_cfg.get('device', 1),
                mono_cfg.get('width', 1600),
                mono_cfg.get('height', 1300),
                mono_cfg.get('fps', 30),
                f"{args.hand.upper()}_MONO"
            )
        
        # 建议
        print(f"\n{'='*70}")
        print("💡 调试建议")
        print('='*70)
        print("""
如果参数不匹配，可能的原因：

1. USB 带宽不足（最常见）
   - 两个摄像头总带宽超过 350 Mbps
   - 解决：降低分辨率或帧率

2. 硬件不支持
   - 摄像头固定某个分辨率或帧率
   - 使用 --list-formats-ext 查看支持的格式

3. 驱动问题
   - 运行 diagnose_usb_bandwidth.py 获取详细信息
   - 检查 lsusb 输出摄像头是否被正确识别

建议的修复步骤：
1. 分别测试每个摄像头（--stereo-only 或 --mono-only）
2. 逐个降低分辨率/帧率，找到最高稳定配置
3. 修改 config.yaml 使用这些稳定配置
4. 使用真正的采集程序验证

示例：
  python3 debug_camera_params.py --hand right --stereo-only
  python3 debug_camera_params.py --hand right --mono-only
""")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ 错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

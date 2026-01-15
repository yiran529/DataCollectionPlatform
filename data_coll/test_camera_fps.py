#!/usr/bin/env python3
"""
相机帧率诊断工具
用于树莓派上排查帧率问题
"""

import cv2
import time
import argparse
import subprocess


def get_v4l2_formats(device_id: int):
    """获取V4L2支持的格式"""
    try:
        result = subprocess.run(
            ['v4l2-ctl', '-d', f'/dev/video{device_id}', '--list-formats-ext'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout
    except:
        return "(无法获取)"


def test_camera(device_id: int, width: int, height: int, fps: int, 
                duration: float = 5.0, use_mjpg: bool = True, use_v4l2: bool = True):
    """测试单个相机帧率"""
    
    print(f"\n{'='*60}")
    print(f"测试设备 /dev/video{device_id}")
    print(f"目标: {width}x{height} @ {fps}fps")
    print(f"MJPG: {'是' if use_mjpg else '否'}, V4L2后端: {'是' if use_v4l2 else '否'}")
    print(f"{'='*60}")
    
    # 打开相机
    if use_v4l2:
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print("❌ 无法打开相机")
        return None
    
    # 设置格式
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # 检查实际参数
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    
    print(f"实际: {actual_w}x{actual_h} @ {actual_fps:.0f}fps ({fourcc_str})")
    
    # 预热
    print("预热...")
    for _ in range(10):
        cap.read()
    
    # 测试帧率
    print(f"采样 {duration}s...")
    frame_count = 0
    intervals = []
    last_time = time.time()
    start_time = time.time()
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        now = time.time()
        
        if ret:
            frame_count += 1
            interval = (now - last_time) * 1000
            intervals.append(interval)
            last_time = now
            
            if frame_count % 30 == 0:
                recent_fps = 1000.0 / (sum(intervals[-30:]) / 30) if intervals else 0
                print(f"\r  帧 {frame_count}: 当前 {recent_fps:.1f} fps", end="", flush=True)
    
    cap.release()
    
    # 统计
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    
    if intervals:
        import numpy as np
        intervals = np.array(intervals[1:])  # 去掉第一个
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        min_interval = np.min(intervals)
        max_interval = np.max(intervals)
        
        print(f"\n\n结果:")
        print(f"  总帧数: {frame_count}")
        print(f"  平均帧率: {avg_fps:.2f} fps")
        print(f"  帧间隔: {avg_interval:.1f} ± {std_interval:.1f} ms")
        print(f"  范围: {min_interval:.1f} ~ {max_interval:.1f} ms")
        
        # 判断
        if avg_fps >= fps * 0.9:
            print(f"  ✅ 达标 (目标 {fps} fps)")
        elif avg_fps >= fps * 0.5:
            print(f"  ⚠️ 偏低 (目标 {fps} fps)")
        else:
            print(f"  ❌ 严重不足 (目标 {fps} fps)")
    
    return avg_fps


def main():
    parser = argparse.ArgumentParser(description="相机帧率诊断")
    parser.add_argument("--device", "-d", type=int, default=0, help="设备ID")
    parser.add_argument("--width", "-W", type=int, default=3840, help="宽度")
    parser.add_argument("--height", "-H", type=int, default=1080, help="高度")
    parser.add_argument("--fps", "-f", type=int, default=60, help="目标帧率")
    parser.add_argument("--duration", "-t", type=float, default=5.0, help="测试时长")
    parser.add_argument("--no-mjpg", action="store_true", help="不使用MJPG")
    parser.add_argument("--no-v4l2", action="store_true", help="不使用V4L2后端")
    parser.add_argument("--scan", action="store_true", help="扫描所有分辨率")
    parser.add_argument("--list", action="store_true", help="列出支持的格式")
    args = parser.parse_args()
    
    print("=" * 60)
    print("相机帧率诊断工具")
    print("=" * 60)
    
    # 列出V4L2格式
    if args.list:
        print(f"\n设备 /dev/video{args.device} 支持的格式:")
        print(get_v4l2_formats(args.device))
        return
    
    # 扫描模式
    if args.scan:
        print(f"\n扫描设备 {args.device} 的最佳设置...")
        
        configs = [
            # (width, height, fps, mjpg)
            (3840, 1080, 60, True),
            (3840, 1080, 30, True),
            (2560, 720, 60, True),
            (2560, 720, 30, True),
            (1920, 540, 60, True),
            (1920, 540, 30, True),
            (1280, 480, 60, True),
            (1280, 480, 30, True),
            # 非MJPG对比
            (1280, 480, 30, False),
        ]
        
        results = []
        for w, h, f, mjpg in configs:
            fps_result = test_camera(args.device, w, h, f, 3.0, mjpg, not args.no_v4l2)
            if fps_result:
                results.append((w, h, f, mjpg, fps_result))
        
        print("\n" + "=" * 60)
        print("扫描结果汇总")
        print("=" * 60)
        print(f"{'分辨率':>12} {'目标FPS':>8} {'MJPG':>6} {'实际FPS':>10} {'状态':>8}")
        print("-" * 60)
        for w, h, f, mjpg, actual in results:
            status = "✅" if actual >= f * 0.9 else "⚠️" if actual >= f * 0.5 else "❌"
            print(f"{w}x{h:>4} {f:>8} {'是' if mjpg else '否':>6} {actual:>10.1f} {status:>8}")
        
        return
    
    # 单次测试
    test_camera(
        args.device, args.width, args.height, args.fps,
        args.duration, not args.no_mjpg, not args.no_v4l2
    )


if __name__ == "__main__":
    main()






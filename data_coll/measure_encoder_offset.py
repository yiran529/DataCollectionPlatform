#!/usr/bin/env python3
"""
测量编码器与相机的时间偏移
方法: 快速转动编码器，观察相机画面中的运动与编码器读数的时间差
"""

import cv2
import numpy as np
import time
import threading
import argparse
from collections import deque
import glob

try:
    import minimalmodbus
except ImportError:
    exit(1)


class MotionDetector:
    """通过帧差法检测相机中的运动"""
    def __init__(self):
        self.prev_frame = None
        self.threshold = 30
        
    def detect(self, frame) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return 0
        
        diff = cv2.absdiff(self.prev_frame, gray)
        motion = np.mean(diff)
        self.prev_frame = gray
        return motion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=6)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--encoder-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--duration", type=int, default=30)
    args = parser.parse_args()
    
    # 相机
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    for _ in range(10):
        cap.read()
    
    # 编码器
    ports = sorted(glob.glob('/dev/ttyUSB*'))
    port = args.encoder_port if args.encoder_port in ports else (ports[0] if ports else None)
    if not port:
        print("No encoder")
        return
    
    inst = minimalmodbus.Instrument(port, 1)
    inst.serial.baudrate = 115200
    inst.serial.timeout = 0.1
    inst.mode = minimalmodbus.MODE_RTU
    inst.clear_buffers_before_each_transaction = True
    
    def read_encoder():
        regs = inst.read_registers(0x40, 2, 3)
        raw = (regs[0] << 16) | regs[1]
        return (raw / 65536) * 360.0 * 0.5 % 360.0
    
    # 测试
    try:
        read_encoder()
        print(f"Encoder OK: {port}")
    except:
        print("Encoder failed")
        return
    
    print("=" * 50)
    print("编码器-相机时间偏移测量")
    print("=" * 50)
    print("操作: 快速转动编码器，程序会检测相机中的运动")
    print("比较编码器变化和相机运动的时间差")
    print("按 q 退出")
    print("=" * 50)
    
    detector = MotionDetector()
    
    # 数据
    encoder_events = []  # (timestamp, angle_change)
    camera_events = []   # (timestamp, motion_amount)
    
    last_angle = read_encoder()
    last_encoder_ts = time.time()
    
    motion_threshold = 5.0
    angle_threshold = 2.0
    
    start = time.time()
    
    while time.time() - start < args.duration:
        # 读编码器
        try:
            angle = read_encoder()
            ts = time.time()
            angle_change = abs(angle - last_angle)
            if angle_change > 180:
                angle_change = 360 - angle_change
            
            if angle_change > angle_threshold and ts - last_encoder_ts > 0.1:
                encoder_events.append((ts, angle_change))
                print(f"[ENC] t={ts:.3f} Δ={angle_change:.1f}°")
                last_encoder_ts = ts
            
            last_angle = angle
        except:
            pass
        
        # 读相机
        ret, frame = cap.read()
        if ret:
            ts = time.time()
            motion = detector.detect(frame)
            
            if motion > motion_threshold and (not camera_events or ts - camera_events[-1][0] > 0.1):
                camera_events.append((ts, motion))
                print(f"[CAM] t={ts:.3f} motion={motion:.1f}")
            
            # 显示
            cv2.putText(frame, f"Encoder events: {len(encoder_events)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Camera events: {len(camera_events)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Offset Measure", cv2.resize(frame, (960, 540)))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    inst.serial.close()
    cv2.destroyAllWindows()
    
    # 分析
    print("\n" + "=" * 50)
    print("分析结果")
    print("=" * 50)
    
    if len(encoder_events) < 2 or len(camera_events) < 2:
        print("数据不足，请多次快速转动编码器")
        return
    
    # 匹配事件，计算时间差
    offsets = []
    for enc_ts, _ in encoder_events:
        # 找最近的相机事件
        best_diff = float('inf')
        for cam_ts, _ in camera_events:
            diff = cam_ts - enc_ts
            if abs(diff) < abs(best_diff) and abs(diff) < 0.5:
                best_diff = diff
        if abs(best_diff) < 0.5:
            offsets.append(best_diff * 1000)  # ms
    
    if offsets:
        print(f"匹配事件数: {len(offsets)}")
        print(f"平均偏移: {np.mean(offsets):.1f} ms")
        print(f"标准差: {np.std(offsets):.1f} ms")
        print(f"范围: {np.min(offsets):.1f} ~ {np.max(offsets):.1f} ms")
        print(f"\n建议 stereo-encoder-offset: {np.mean(offsets):.1f} ms")
    else:
        print("无法匹配事件，请确保编码器运动被相机捕获")


if __name__ == "__main__":
    main()















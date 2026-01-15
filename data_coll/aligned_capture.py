#!/usr/bin/env python3
"""
同步采集并对齐: 双目相机 + 单目相机 + 角度编码器
输出: 时间对齐的图像序列和角度数据

特性:
- 预热阶段: 等待帧率稳定
- 自动校准: 测量稳定后的传感器间偏移
- 使用实测偏移进行对齐
"""

import cv2
import numpy as np
import time
import os
import threading
import argparse
from queue import Queue
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque
import json
import glob

try:
    import minimalmodbus
except ImportError:
    print("pip install minimalmodbus")
    exit(1)


@dataclass
class SensorData:
    timestamp: float
    data: any
    idx: int


class CameraCapture:
    def __init__(self, device_id: int, width: int, height: int, fps: int, name: str):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name
        self.cap = None
        self.buffer = []
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.count = 0
        self.start_time = 0
        self.recent_intervals = deque(maxlen=30)  # 最近30帧的间隔
        self.last_ts = 0
        
    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        for _ in range(10):
            self.cap.read()
        print(f"[{self.name}] {self.device_id}: {self.width}x{self.height} @ {self.cap.get(cv2.CAP_PROP_FPS):.0f}fps")
        return True
    
    def start(self):
        self.running = True
        self.buffer = []
        self.count = 0
        self.start_time = time.time()
        self.last_ts = 0
        self.recent_intervals.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
    
    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            ts = time.time()
            if ret:
                self.count += 1
                # 计算帧间隔
                if self.last_ts > 0:
                    interval = (ts - self.last_ts) * 1000
                    self.recent_intervals.append(interval)
                self.last_ts = ts
                
                with self.lock:
                    self.buffer.append(SensorData(ts, frame.copy(), self.count))
                    if len(self.buffer) > 300:
                        self.buffer.pop(0)
    
    def get_all(self) -> List[SensorData]:
        with self.lock:
            data = self.buffer.copy()
            self.buffer = []
        return data
    
    def get_latest(self) -> Optional[SensorData]:
        with self.lock:
            if self.buffer:
                return self.buffer[-1]
        return None
    
    def get_fps(self) -> float:
        if len(self.recent_intervals) < 5:
            return 0
        return 1000.0 / np.mean(self.recent_intervals)
    
    def get_fps_std(self) -> float:
        if len(self.recent_intervals) < 5:
            return 999
        intervals = np.array(self.recent_intervals)
        return np.std(intervals)


class EncoderCapture:
    REG_ANGLE = 0x40
    
    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.inst = None
        self.buffer = []
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.count = 0
        
    def open(self) -> bool:
        try:
            self.inst = minimalmodbus.Instrument(self.port, 1)
            self.inst.serial.baudrate = self.baudrate
            self.inst.serial.timeout = 0.1
            self.inst.mode = minimalmodbus.MODE_RTU
            self.inst.clear_buffers_before_each_transaction = True
            self._read()
            print(f"[ENCODER] {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"[ENCODER] Failed: {e}")
            return False
    
    def _read(self) -> float:
        regs = self.inst.read_registers(self.REG_ANGLE, 2, 3)
        raw = (regs[0] << 16) | regs[1]
        return (raw / 65536) * 360.0 * 0.5 % 360.0  # 0.5 = scale factor
    
    def start(self):
        self.running = True
        self.buffer = []
        self.count = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.inst:
            self.inst.serial.close()
    
    def _loop(self):
        while self.running:
            try:
                angle = self._read()
                ts = time.time()
                self.count += 1
                with self.lock:
                    self.buffer.append(SensorData(ts, angle, self.count))
                    if len(self.buffer) > 500:
                        self.buffer.pop(0)
            except:
                pass
    
    def get_all(self) -> List[SensorData]:
        with self.lock:
            data = self.buffer.copy()
            self.buffer = []
        return data
    
    def get_latest(self) -> Optional[SensorData]:
        with self.lock:
            if self.buffer:
                return self.buffer[-1]
        return None


def find_nearest(target_ts: float, data_list: List[SensorData], max_diff_ms: float = 50) -> Optional[SensorData]:
    """找到最接近目标时间戳的数据"""
    if not data_list:
        return None
    best = min(data_list, key=lambda x: abs(x.timestamp - target_ts))
    if abs(best.timestamp - target_ts) * 1000 <= max_diff_ms:
        return best
    return None


def warmup_and_calibrate(stereo: 'CameraCapture', mono: 'CameraCapture', 
                         encoder: Optional['EncoderCapture'],
                         target_fps: float = 30.0,
                         warmup_time: float = 7.0,
                         calibration_time: float = 3.0) -> Tuple[float, float]:
    """
    预热并精确校准偏移量
    
    流程:
    1. 预热阶段: 固定时间等待相机稳定
    2. 精确校准: 采集样本计算精确偏移
    
    总时间约 warmup_time + calibration_time 秒
    
    Returns:
        (stereo_mono_offset_ms, stereo_encoder_offset_ms)
    """
    print("\n" + "=" * 60)
    print(f"阶段1: 预热 {warmup_time}s - 等待相机稳定...")
    print("=" * 60)
    
    start = time.time()
    
    # 固定时间预热
    while time.time() - start < warmup_time:
        elapsed = time.time() - start
        s_fps = stereo.get_fps()
        m_fps = mono.get_fps()
        s_std = stereo.get_fps_std()
        m_std = mono.get_fps_std()
        
        print(f"\r[{elapsed:.1f}s/{warmup_time}s] "
              f"S:{s_fps:.1f}fps(±{s_std:.1f}) M:{m_fps:.1f}fps(±{m_std:.1f})", end="", flush=True)
        time.sleep(0.1)
    
    print(f"\n✓ 预热完成")
    
    # 清空缓冲区
    stereo.get_all()
    mono.get_all()
    if encoder:
        encoder.get_all()
    
    print(f"\n" + "=" * 60)
    print(f"阶段2: 精确校准 - 采集 {calibration_time}s 数据...")
    print("=" * 60)
    
    # 精确校准阶段 - 采集配对的帧来计算偏移
    calib_start = time.time()
    paired_offsets_sm = []  # (stereo_ts, mono_ts, offset)
    paired_offsets_se = []  # (stereo_ts, encoder_ts, offset)
    
    stereo_frames = []
    mono_frames = []
    encoder_samples = []
    
    while time.time() - calib_start < calibration_time:
        time.sleep(0.05)
        
        # 收集数据
        stereo_frames.extend(stereo.get_all())
        mono_frames.extend(mono.get_all())
        if encoder:
            encoder_samples.extend(encoder.get_all())
        
        elapsed = time.time() - calib_start
        print(f"\r  采集中... S:{len(stereo_frames)} M:{len(mono_frames)} E:{len(encoder_samples)}", end="", flush=True)
    
    print(f"\n  共采集 S:{len(stereo_frames)} M:{len(mono_frames)} E:{len(encoder_samples)}")
    
    # 配对计算偏移: 为每个stereo帧找最近的mono帧
    for s in stereo_frames:
        # 找最近的mono帧
        best_mono = None
        best_diff = float('inf')
        for m in mono_frames:
            diff = abs(s.timestamp - m.timestamp)
            if diff < best_diff:
                best_diff = diff
                best_mono = m
        
        if best_mono and best_diff < 0.05:  # 50ms内
            offset = (s.timestamp - best_mono.timestamp) * 1000
            paired_offsets_sm.append(offset)
        
        # 找最近的encoder
        if encoder_samples:
            best_enc = None
            best_diff = float('inf')
            for e in encoder_samples:
                diff = abs(s.timestamp - e.timestamp)
                if diff < best_diff:
                    best_diff = diff
                    best_enc = e
            
            if best_enc and best_diff < 0.05:
                offset = (s.timestamp - best_enc.timestamp) * 1000
                paired_offsets_se.append(offset)
    
    # 计算偏移统计
    if paired_offsets_sm:
        # 去掉最大最小10%的异常值
        sorted_sm = sorted(paired_offsets_sm)
        trim = max(1, len(sorted_sm) // 10)
        trimmed_sm = sorted_sm[trim:-trim] if len(sorted_sm) > trim * 2 else sorted_sm
        
        final_sm = np.mean(trimmed_sm)
        sm_std = np.std(trimmed_sm)
        sm_median = np.median(trimmed_sm)
    else:
        final_sm = 0.0
        sm_std = 0.0
        sm_median = 0.0
    
    if paired_offsets_se:
        sorted_se = sorted(paired_offsets_se)
        trim = max(1, len(sorted_se) // 10)
        trimmed_se = sorted_se[trim:-trim] if len(sorted_se) > trim * 2 else sorted_se
        
        final_se = np.mean(trimmed_se)
        se_std = np.std(trimmed_se)
    else:
        final_se = 0.0
        se_std = 0.0
    
    print(f"\n" + "=" * 60)
    print(f"✓ 校准完成:")
    print(f"  Stereo-Mono偏移:")
    print(f"    均值: {final_sm:+.2f} ms")
    print(f"    中位数: {sm_median:+.2f} ms")
    print(f"    标准差: {sm_std:.2f} ms")
    print(f"    样本数: {len(paired_offsets_sm)}")
    print(f"  Stereo-Encoder偏移: {final_se:+.2f} ms (±{se_std:.2f})")
    print(f"  当前帧率: S={stereo.get_fps():.1f}fps, M={mono.get_fps():.1f}fps")
    print("=" * 60)
    
    # 使用中位数更稳定
    return sm_median, final_se


def align_data(stereo_data: List[SensorData], 
               mono_data: List[SensorData], 
               encoder_data: List[SensorData],
               stereo_mono_offset_ms: float = 35.0,
               stereo_encoder_offset_ms: float = 0.0) -> List[Tuple]:
    """
    对齐三类数据
    以双目相机为参考，找到对应的单目和编码器数据
    
    offset: 正值表示stereo比该传感器晚
    """
    aligned = []
    
    for stereo in stereo_data:
        # 单目: stereo_ts - offset = mono_ts (offset正值表示stereo晚)
        mono_target_ts = stereo.timestamp - stereo_mono_offset_ms / 1000.0
        mono = find_nearest(mono_target_ts, mono_data)
        
        # 编码器: stereo_ts - offset = encoder_ts
        encoder_target_ts = stereo.timestamp - stereo_encoder_offset_ms / 1000.0
        encoder = find_nearest(encoder_target_ts, encoder_data)
        
        if mono and encoder:
            aligned.append((stereo, mono, encoder))
    
    return aligned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stereo-device", type=int, default=6)
    parser.add_argument("--stereo-width", type=int, default=3840)
    parser.add_argument("--stereo-height", type=int, default=1080)
    parser.add_argument("--mono-device", type=int, default=4)
    parser.add_argument("--mono-width", type=int, default=1280)
    parser.add_argument("--mono-height", type=int, default=1024)
    parser.add_argument("--encoder-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--warmup-time", type=float, default=7.0, help="预热时间(秒)")
    parser.add_argument("--calib-time", type=float, default=3.0, help="校准采样时间(秒)")
    parser.add_argument("--manual-offset-sm", type=float, default=None, help="手动指定S-M偏移(跳过自动校准)")
    parser.add_argument("--manual-offset-se", type=float, default=None, help="手动指定S-E偏移(跳过自动校准)")
    parser.add_argument("--duration", "-t", type=int, default=10)
    parser.add_argument("--output", "-o", type=str, default="aligned_data")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save-video", "-v", action="store_true", help="保存拼接视频")
    parser.add_argument("--no-images", action="store_true", help="不保存单独图像")
    args = parser.parse_args()
    
    # 创建输出目录
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output, f"capture_{ts}")
    stereo_dir = os.path.join(out_dir, "stereo")
    mono_dir = os.path.join(out_dir, "mono")
    os.makedirs(stereo_dir, exist_ok=True)
    os.makedirs(mono_dir, exist_ok=True)
    
    print("=" * 50)
    print("同步采集: 双目 + 单目 + 编码器")
    print("=" * 50)
    
    # 初始化设备
    stereo = CameraCapture(args.stereo_device, args.stereo_width, args.stereo_height, args.fps, "STEREO")
    mono = CameraCapture(args.mono_device, args.mono_width, args.mono_height, args.fps, "MONO")
    
    ports = sorted(glob.glob('/dev/ttyUSB*'))
    encoder_port = args.encoder_port if args.encoder_port in ports else (ports[0] if ports else None)
    encoder = EncoderCapture(encoder_port) if encoder_port else None
    
    if not stereo.open() or not mono.open():
        return
    use_encoder = encoder and encoder.open()
    
    # 启动设备
    stereo.start()
    mono.start()
    if use_encoder:
        encoder.start()
    
    # 预热并校准偏移
    if args.manual_offset_sm is not None and args.manual_offset_se is not None:
        sm_offset = args.manual_offset_sm
        se_offset = args.manual_offset_se
        print(f"\n使用手动指定偏移: S-M={sm_offset}ms, S-E={se_offset}ms")
        time.sleep(1.0)  # 简单预热
    else:
        sm_offset, se_offset = warmup_and_calibrate(
            stereo, mono, encoder if use_encoder else None,
            target_fps=args.fps,
            warmup_time=args.warmup_time,
            calibration_time=args.calib_time
        )
    
    # 清空预热期间的数据
    stereo.get_all()
    mono.get_all()
    if use_encoder:
        encoder.get_all()
    
    # 开始正式采集
    print(f"\n开始采集 {args.duration} 秒...")
    start = time.time()
    
    all_stereo = []
    all_mono = []
    all_encoder = []
    
    try:
        while time.time() - start < args.duration:
            time.sleep(0.1)
            all_stereo.extend(stereo.get_all())
            all_mono.extend(mono.get_all())
            if use_encoder:
                all_encoder.extend(encoder.get_all())
            
            elapsed = time.time() - start
            print(f"\r[{elapsed:.1f}s] S:{len(all_stereo)} M:{len(all_mono)} E:{len(all_encoder)}", end="", flush=True)
    
    except KeyboardInterrupt:
        pass
    
    stereo.stop()
    mono.stop()
    if use_encoder:
        encoder.stop()
    
    print(f"\n\n采集完成: S={len(all_stereo)}, M={len(all_mono)}, E={len(all_encoder)}")
    
    # 对齐（使用校准得到的偏移）
    print("\n对齐数据...")
    aligned = align_data(all_stereo, all_mono, all_encoder, sm_offset, se_offset)
    
    print(f"对齐帧数: {len(aligned)}")
    
    if not aligned:
        print("无对齐数据")
        return
    
    # 降采样到目标fps
    target_interval = 1.0 / args.fps
    filtered = [aligned[0]]
    for item in aligned[1:]:
        if item[0].timestamp - filtered[-1][0].timestamp >= target_interval * 0.9:
            filtered.append(item)
    
    print(f"输出帧数: {len(filtered)} @ {args.fps}fps")
    
    # 保存
    print("\n保存...")
    angles = []
    
    # 准备视频写入器
    video_writer = None
    if args.save_video:
        # 计算拼接后的视频尺寸
        # 布局: [左眼 | 右眼 | 单目(缩放)]
        stereo_h = args.stereo_height
        stereo_w = args.stereo_width // 2  # 每个眼的宽度
        mono_h, mono_w = args.mono_height, args.mono_width
        
        # 缩放单目到与双目同高
        mono_scale = stereo_h / mono_h
        scaled_mono_w = int(mono_w * mono_scale)
        
        video_w = stereo_w * 2 + scaled_mono_w
        video_h = stereo_h
        
        video_path = os.path.join(out_dir, "combined.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, args.fps, (video_w, video_h))
        print(f"  视频尺寸: {video_w}x{video_h}")
    
    for i, (s, m, e) in enumerate(filtered):
        idx = i + 1
        # 双目图像 (分离左右)
        h, w = s.data.shape[:2]
        left = s.data[:, :w//2]
        right = s.data[:, w//2:]
        
        # 保存单独图像
        if not args.no_images:
            cv2.imwrite(os.path.join(stereo_dir, f"{idx:06d}_L.jpg"), left, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(stereo_dir, f"{idx:06d}_R.jpg"), right, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(mono_dir, f"{idx:06d}.jpg"), m.data, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # 写入视频
        if video_writer:
            # 缩放单目图像到与双目同高
            mono_scaled = cv2.resize(m.data, (scaled_mono_w, stereo_h))
            # 拼接: [左眼 | 右眼 | 单目]
            combined = np.hstack([left, right, mono_scaled])
            
            # 添加角度信息
            angle_text = f"Angle: {e.data:.1f}deg"
            cv2.putText(combined, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(combined, f"Frame: {idx}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            video_writer.write(combined)
        
        # 角度
        angles.append({
            "frame": idx,
            "timestamp": s.timestamp,
            "angle": e.data,
            "stereo_ts": s.timestamp,
            "mono_ts": m.timestamp,
            "encoder_ts": e.timestamp
        })
        
        if idx % 30 == 0:
            print(f"\r  {idx}/{len(filtered)}", end="", flush=True)
    
    if video_writer:
        video_writer.release()
        print(f"\n  视频已保存: combined.mp4")
    
    # 保存角度数据
    with open(os.path.join(out_dir, "angles.json"), 'w') as f:
        json.dump(angles, f, indent=2)
    
    # 保存配置
    config = {
        "fps": args.fps,
        "total_frames": len(filtered),
        "stereo_mono_offset_ms": sm_offset,
        "stereo_encoder_offset_ms": se_offset,
        "stereo_size": [args.stereo_width // 2, args.stereo_height],
        "mono_size": [args.mono_width, args.mono_height]
    }
    with open(os.path.join(out_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n\n完成! 输出: {out_dir}")
    if not args.no_images:
        print(f"  stereo/: {len(filtered)} 对 (左右分离)")
        print(f"  mono/: {len(filtered)} 张")
    if args.save_video:
        print(f"  combined.mp4: {len(filtered)} 帧 @ {args.fps}fps")
    print(f"  angles.json: {len(filtered)} 条")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
高同步数据收集类
同步采集: 双目相机 + 单目相机 + 角度编码器

使用方式:
    collector = SyncDataCollector("config.yaml")
    collector.start()
    collector.wait_ready()
    
    collector.start_recording()
    ...
    data = collector.stop_recording()
    
    collector.stop()
"""

import cv2
import numpy as np
import time
import threading
import os
import json
import yaml
import glob
from dataclasses import dataclass
from typing import Optional, List, Dict
from collections import deque

try:
    import minimalmodbus
except ImportError:
    minimalmodbus = None

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


@dataclass
class SensorFrame:
    """传感器帧数据"""
    timestamp: float
    data: any
    idx: int


@dataclass
class SyncFrame:
    """同步帧（三个传感器对齐后）"""
    stereo: np.ndarray
    mono: np.ndarray
    angle: float
    timestamp: float
    stereo_ts: float
    mono_ts: float
    encoder_ts: float
    idx: int


class Config:
    """配置管理"""
    
    def __init__(self, yaml_path: str = None):
        # 默认值
        self.stereo_device = 6
        self.stereo_width = 3840
        self.stereo_height = 1080
        self.stereo_fps = 30
        
        self.mono_device = 4
        self.mono_width = 1280
        self.mono_height = 1024
        self.mono_fps = 30
        
        self.encoder_port = "/dev/ttyUSB0"
        self.encoder_baudrate = 115200
        self.encoder_calibration_file = ""
        
        self.warmup_time = 7.0
        self.calib_time = 3.0
        self.max_time_diff_ms = 50.0
        
        self.save_output_dir = "./data"
        self.save_jpeg_quality = 85
        
        self._config_dir = ""
        
        if yaml_path:
            self.load(yaml_path)
    
    def load(self, yaml_path: str):
        """从YAML加载配置"""
        self._config_dir = os.path.dirname(os.path.abspath(yaml_path))
        
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # 双目相机
        if 'stereo' in cfg:
            s = cfg['stereo']
            self.stereo_device = s.get('device', self.stereo_device)
            self.stereo_width = s.get('width', self.stereo_width)
            self.stereo_height = s.get('height', self.stereo_height)
            self.stereo_fps = s.get('fps', self.stereo_fps)
        
        # 单目相机
        if 'mono' in cfg:
            m = cfg['mono']
            self.mono_device = m.get('device', self.mono_device)
            self.mono_width = m.get('width', self.mono_width)
            self.mono_height = m.get('height', self.mono_height)
            self.mono_fps = m.get('fps', self.mono_fps)
        
        # 编码器
        if 'encoder' in cfg:
            e = cfg['encoder']
            self.encoder_port = e.get('port', self.encoder_port)
            self.encoder_baudrate = e.get('baudrate', self.encoder_baudrate)
            calib_file = e.get('calibration_file', '')
            if calib_file:
                self.encoder_calibration_file = os.path.join(self._config_dir, calib_file)
        
        # 预热校准
        self.warmup_time = cfg.get('warmup_time', self.warmup_time)
        self.calib_time = cfg.get('calib_time', self.calib_time)
        self.max_time_diff_ms = cfg.get('max_time_diff_ms', self.max_time_diff_ms)
        
        # 保存
        if 'save' in cfg:
            sv = cfg['save']
            self.save_output_dir = sv.get('output_dir', self.save_output_dir)
            self.save_jpeg_quality = sv.get('jpeg_quality', self.save_jpeg_quality)
        
        print(f"[CONFIG] 已加载: {yaml_path}")


class CameraReader:
    """摄像头读取器"""
    
    def __init__(self, device_id: int, width: int, height: int, fps: int, name: str):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name
        
        self.cap = None
        self.buffer: deque = deque(maxlen=300)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.count = 0
        self.recent_intervals: deque = deque(maxlen=30)
        self.last_ts = 0
        
    def open(self) -> bool:
        # 使用默认后端（与 aligned_capture.py 相同）
        self.cap = cv2.VideoCapture(self.device_id)
        
        if not self.cap.isOpened():
            print(f"[{self.name}] 无法打开设备 {self.device_id}")
            return False
        
        # 设置 MJPG 格式
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # 预热
        for _ in range(10):
            self.cap.read()
        
        # 检查实际参数
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[{self.name}] {self.device_id}: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")
        return True
    
    def start(self):
        self.running = True
        self.count = 0
        self.last_ts = 0
        self.buffer.clear()
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
                if self.last_ts > 0:
                    self.recent_intervals.append((ts - self.last_ts) * 1000)
                self.last_ts = ts
                with self.lock:
                    self.buffer.append(SensorFrame(ts, frame.copy(), self.count))
    
    def get_buffer(self) -> List[SensorFrame]:
        with self.lock:
            return list(self.buffer)
    
    def clear_buffer(self):
        with self.lock:
            self.buffer.clear()
    
    def get_fps(self) -> float:
        if len(self.recent_intervals) < 5:
            return 0
        return 1000.0 / np.mean(self.recent_intervals)
    
    def get_fps_std(self) -> float:
        if len(self.recent_intervals) < 5:
            return 999
        return np.std(list(self.recent_intervals))


class EncoderReader:
    """编码器读取器（带校准）"""
    REG_ANGLE = 0x40
    
    def __init__(self, port: str, baudrate: int, calibration_file: str = ""):
        self.port = port
        self.baudrate = baudrate
        self.calibration_file = calibration_file
        
        self.inst = None
        self.buffer: deque = deque(maxlen=500)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.count = 0
        
        # 校准参数
        self.angle_zero = 0.0
        self.calibrated = False
        self.last_raw_angle = 0.0
        self.accumulated_turns = 0
    
    def _load_calibration(self):
        """加载校准配置"""
        if not self.calibration_file or not os.path.exists(self.calibration_file):
            return
        try:
            with open(self.calibration_file, 'r') as f:
                cfg = json.load(f)
            self.angle_zero = cfg.get("angle_zero", 0.0)
            self.calibrated = cfg.get("calibrated", False)
            if self.calibrated:
                print(f"[ENCODER] 校准零点: {self.angle_zero:.2f}°")
        except Exception as e:
            print(f"[ENCODER] 加载校准失败: {e}")
    
    def open(self) -> bool:
        if not minimalmodbus:
            print("[ENCODER] minimalmodbus 未安装")
            return False
        
        # 检查串口是否存在
        if not os.path.exists(self.port):
            print(f"[ENCODER] 串口不存在: {self.port}")
            return False
        
        # 检查串口权限
        if not os.access(self.port, os.R_OK | os.W_OK):
            print(f"[ENCODER] 串口权限不足: {self.port}")
            print(f"[ENCODER] 尝试: sudo chmod 666 {self.port} 或添加用户到dialout组")
            return False
        
        try:
            self.inst = minimalmodbus.Instrument(self.port, 1)
            self.inst.serial.baudrate = self.baudrate
            self.inst.serial.bytesize = 8
            self.inst.serial.parity = minimalmodbus.serial.PARITY_NONE
            self.inst.serial.stopbits = 1
            self.inst.serial.timeout = 0.5  # 增加超时时间，与测试脚本一致
            self.inst.mode = minimalmodbus.MODE_RTU
            self.inst.clear_buffers_before_each_transaction = True
            
            self._load_calibration()
            
            # 尝试读取一次角度值来验证连接
            try:
                raw = self._read_raw()
                self.last_raw_angle = raw if self.calibrated else self.angle_zero
                print(f"[ENCODER] {self.port} @ {self.baudrate} - 连接成功")
                return True
            except Exception as read_e:
                print(f"[ENCODER] 连接成功但读取失败: {read_e}")
                print(f"[ENCODER] 可能是设备地址或寄存器地址不匹配，继续尝试...")
                # 即使读取失败，也返回True，让程序继续运行
                self.last_raw_angle = self.angle_zero
            return True
        except Exception as e:
            print(f"[ENCODER] 连接失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _read_raw(self) -> float:
        """读取原始角度"""
        regs = self.inst.read_registers(self.REG_ANGLE, 2, 3)
        raw = (regs[0] << 16) | regs[1]
        return (raw / 65536) * 360.0 * 2.0 % 360.0
    
    def _get_calibrated_angle(self, raw_angle: float) -> float:
        """计算校准后的角度"""
        if not self.calibrated:
            return raw_angle
        
        # 检测 0/360 边界
        delta = raw_angle - self.last_raw_angle
        if delta > 180:
            self.accumulated_turns -= 1
        elif delta < -180:
            self.accumulated_turns += 1
        
        self.last_raw_angle = raw_angle
        
        angle_diff = raw_angle - self.angle_zero
        total_diff = self.accumulated_turns * 360.0 + angle_diff
        
        return -total_diff
    
    def start(self):
        self.running = True
        self.buffer.clear()
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
                raw = self._read_raw()
                angle = self._get_calibrated_angle(raw)
                ts = time.time()
                self.count += 1
                with self.lock:
                    self.buffer.append(SensorFrame(ts, angle, self.count))
            except Exception as e:
                # 只在调试时打印错误，避免刷屏
                if self.count == 0 or self.count % 100 == 0:
                    print(f"[ENCODER] 读取错误 (第{self.count}次): {e}")
                pass
    
    def get_buffer(self) -> List[SensorFrame]:
        with self.lock:
            return list(self.buffer)
    
    def clear_buffer(self):
        with self.lock:
            self.buffer.clear()
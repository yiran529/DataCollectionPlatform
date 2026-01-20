#!/usr/bin/env python3
"""
双手同步数据收集器

同时收集左右手的数据：
- 左手：单目相机 + 双目相机 + 角度编码器
- 右手：单目相机 + 双目相机 + 角度编码器

所有数据保证时间同步对齐
"""

import sys
import os
import cv2
import numpy as np
import time
import threading
import yaml
import json
import glob
from dataclasses import dataclass
from typing import Optional, List, Dict
from collections import deque

# 添加父目录到路径，以便导入sync_data_collector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_coll.sync_data_collector import CameraReader, EncoderReader, SensorFrame

try:
    import minimalmodbus
except ImportError:
    minimalmodbus = None

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class HandFrame:
    """单手的同步帧"""
    stereo: np.ndarray
    mono: np.ndarray
    angle: float
    timestamp: float
    stereo_ts: float
    mono_ts: float
    encoder_ts: float
    idx: int


@dataclass
class DualHandFrame:
    """双手同步帧"""
    left: HandFrame
    right: HandFrame
    timestamp: float
    idx: int


class HandCollector:
    """单手数据收集器（复用单手的同步逻辑）"""
    
    def __init__(self, config: dict, hand_name: str):
        self.config = config
        self.hand_name = hand_name
        
        self.stereo: Optional[CameraReader] = None
        self.mono: Optional[CameraReader] = None
        self.encoder: Optional[EncoderReader] = None
        
        self._ready = False
        self._running = False
        self._recording = False
        
        # 同步偏移量
        self.stereo_mono_offset_ms = 0.0
        self.stereo_encoder_offset_ms = 0.0
        
        # 立体校正参数
        self.stereo_rectify_enabled = False
        self.stereo_map1_l = None
        self.stereo_map2_l = None
        self.stereo_map1_r = None
        self.stereo_map2_r = None
        self.stereo_calibration = None
        
        # 录制数据（使用deque限制大小，自动丢弃旧数据）
        # 最大录制帧数：默认1800帧（30fps*60秒）
        max_frames = config.get('max_record_frames', 1800)
        self._max_record_frames = max_frames
        self._record_start_ts = 0
        self._recorded_stereo: deque = deque(maxlen=max_frames)
        self._recorded_mono: deque = deque(maxlen=max_frames)
        self._recorded_encoder: deque = deque(maxlen=max_frames)
        self._record_lock = threading.Lock()
        self._record_thread = None
        self._record_stats = {'frames': 0, 'last_print': 0}
        
        # 增量复制：记录上次处理的索引
        self._last_copied_idx = {'stereo': 0, 'mono': 0, 'encoder': 0}
        
        # JPEG压缩质量
        self._jpeg_quality = config.get('jpeg_quality', 85)
    
    @property
    def is_ready(self) -> bool:
        return self._ready
    
    @property
    def is_recording(self) -> bool:
        return self._recording
    
    def start(self) -> bool:
        """启动收集器"""
        if self._running:
            return True
        
        self._running = True
        self._ready = False
        
        thread = threading.Thread(target=self._startup_routine, daemon=True)
        thread.start()
        return True
    
    def wait_ready(self, timeout: float = 30.0) -> bool:
        """等待就绪"""
        start = time.time()
        while not self._ready and (time.time() - start) < timeout:
            time.sleep(0.1)
        return self._ready
    
    def _startup_routine(self):
        """启动流程"""
        cfg = self.config
        
        print(f"\n[{self.hand_name}] 初始化...")
        
        # 初始化相机
        mono_cfg = cfg.get('mono', {})
        stereo_cfg = cfg.get('stereo', {})
        
        # 创建相机对象（但先不打开）
        self.stereo = CameraReader(
            stereo_cfg.get('device', 2),
            stereo_cfg.get('width', 3840),
            stereo_cfg.get('height', 1080),
            stereo_cfg.get('fps', 30),
            f"{self.hand_name}_STEREO"
        )
        
        self.mono = CameraReader(
            mono_cfg.get('device', 0),
            mono_cfg.get('width', 1280),
            mono_cfg.get('height', 1024),
            mono_cfg.get('fps', 30),
            f"{self.hand_name}_MONO"
        )
        
        # 重要：先打开 Stereo 摄像头，再打开 Mono
        # 这样可以避免 USB 竞争条件导致的预热超时
        print(f"[{self.hand_name}] 打开 Stereo 摄像头...")
        if not self.stereo.open():
            print(f"[{self.hand_name}] ❌ Stereo 摄像头初始化失败")
            self._running = False
            return
        
        print(f"[{self.hand_name}] 打开 Mono 摄像头...")
        if not self.mono.open():
            print(f"[{self.hand_name}] ❌ Mono 摄像头初始化失败")
            self.stereo.cap.release()  # 释放 Stereo
            self._running = False
            return
        
        # 加载立体校正参数
        self._load_stereo_rectification()
        
        # 初始化编码器
        encoder_cfg = cfg.get('encoder', {})
        port = encoder_cfg.get('port', '/dev/ttyUSB0')
        baudrate = encoder_cfg.get('baudrate', 115200)
        
        # 从YAML配置读取校准参数（优先），如果没有则尝试从文件读取
        calibration_cfg = encoder_cfg.get('calibration', {})
        calib_file = encoder_cfg.get('calibration_file', '')
        
        # 检查串口是否存在
        ports = sorted(glob.glob('/dev/ttyUSB*'))
        available_ports = ports + sorted(glob.glob('/dev/ttyACM*'))
        
        print(f"[{self.hand_name}] 检查编码器串口: {port}")
        print(f"[{self.hand_name}] 可用串口: {available_ports}")
        
        if port not in available_ports:
            if available_ports:
                print(f"[{self.hand_name}] ⚠️ 配置的串口 {port} 不存在，尝试使用 {available_ports[0]}")
                port = available_ports[0]
            else:
                print(f"[{self.hand_name}] ⚠️ 未找到可用串口，跳过编码器初始化")
                port = None
        
        if port:
            # 检查串口权限
            if not os.access(port, os.R_OK | os.W_OK):
                print(f"[{self.hand_name}] ⚠️ 串口 {port} 权限不足，尝试使用sudo或添加用户到dialout组")
            
            # 创建EncoderReader（使用空字符串作为calibration_file，因为我们将手动设置）
            self.encoder = EncoderReader(port, baudrate, '')
            
            # 如果YAML中有校准配置，直接设置
            if calibration_cfg:
                self.encoder.angle_zero = calibration_cfg.get('angle_zero', 0.0)
                self.encoder.calibrated = calibration_cfg.get('calibrated', False)
                print(f"[{self.hand_name}] 编码器校准参数: angle_zero={self.encoder.angle_zero:.2f}°, calibrated={self.encoder.calibrated}")
            # 否则尝试从文件加载
            elif calib_file:
                # 解析校准文件路径
                if not os.path.isabs(calib_file):
                    # 相对于配置文件目录
                    config_dir = os.path.dirname(os.path.abspath(__file__))
                    calib_file = os.path.join(config_dir, calib_file)
                if os.path.exists(calib_file):
                    self.encoder.calibration_file = calib_file
                    print(f"[{self.hand_name}] 编码器校准文件: {calib_file}")
                else:
                    print(f"[{self.hand_name}] ⚠️ 校准文件不存在: {calib_file}")
        
        use_encoder = False
        if self.encoder:
            print(f"[{self.hand_name}] 尝试连接编码器 {port} @ {baudrate}...")
            use_encoder = self.encoder.open()
            if use_encoder:
                print(f"[{self.hand_name}] ✓ 编码器连接成功")
            else:
                print(f"[{self.hand_name}] ❌ 编码器连接失败")
                # 尝试诊断问题
                if not minimalmodbus:
                    print(f"[{self.hand_name}]   原因: minimalmodbus 未安装")
                elif not os.path.exists(port):
                    print(f"[{self.hand_name}]   原因: 串口 {port} 不存在")
                elif not os.access(port, os.R_OK | os.W_OK):
                    print(f"[{self.hand_name}]   原因: 串口 {port} 权限不足")
                else:
                    print(f"[{self.hand_name}]   原因: 可能是设备地址或波特率不匹配")
        
        # 如果从YAML设置了校准参数，在open之后需要重新设置（因为open会调用_load_calibration）
        if use_encoder and calibration_cfg:
            self.encoder.angle_zero = calibration_cfg.get('angle_zero', 0.0)
            self.encoder.calibrated = calibration_cfg.get('calibrated', False)
            if self.encoder.calibrated:
                print(f"[{self.hand_name}] 编码器校准零点: {self.encoder.angle_zero:.2f}°")
        
        # 启动采集
        self.mono.start()
        self.stereo.start()
        if use_encoder:
            self.encoder.start()
            # 等待一下，确保编码器开始读取
            time.sleep(0.2)
            encoder_test = self.encoder.get_buffer()
            if encoder_test:
                print(f"[{self.hand_name}] ✓ 编码器正在读取数据 ({len(encoder_test)} 条)")
            else:
                print(f"[{self.hand_name}] ⚠️ 编码器已启动，但缓冲区为空（可能正在初始化）")
        
        print(f"[{self.hand_name}] ✓ 相机已启动")
        
        self._ready = True
    
    def _load_stereo_rectification(self):
        """加载立体校正参数（优先从config.yaml读取，否则从JSON文件读取）"""
        stereo_cfg = self.config.get('stereo', {})
        
        # 确定是左手还是右手
        hand_name = self.hand_name.lower()
        is_left = 'left' in hand_name
        hand_key = 'left' if is_left else 'right'
        
        calib = None
        calib_source = None
        
        # 优先从config.yaml读取标定参数
        calibration_cfg = stereo_cfg.get('calibration', {})
        if calibration_cfg and calibration_cfg.get('calibrated', False):
            try:
                # 从YAML配置构建标定字典
                calib = {
                    'left_camera_matrix': calibration_cfg['left_camera_matrix'],
                    'left_distortion': calibration_cfg['left_distortion'],
                    'right_camera_matrix': calibration_cfg['right_camera_matrix'],
                    'right_distortion': calibration_cfg['right_distortion'],
                    'rectify_rotation_left': calibration_cfg['rectify_rotation_left'],
                    'rectify_rotation_right': calibration_cfg['rectify_rotation_right'],
                    'projection_left': calibration_cfg['projection_left'],
                    'projection_right': calibration_cfg['projection_right'],
                    'image_size': calibration_cfg['image_size'],
                    'baseline_mm': calibration_cfg.get('baseline_mm', 0),
                    'reprojection_error': calibration_cfg.get('reprojection_error', 0),
                }
                calib_source = "config.yaml"
            except Exception as e:
                print(f"[{self.hand_name}] ⚠️ 从config.yaml读取标定参数失败: {e}")
                calib = None
        
        # 如果config.yaml中没有，尝试从JSON文件读取
        if calib is None:
            calib_file = stereo_cfg.get('calibration_file', '')
            
            # 如果没有指定，尝试默认路径（根据左右手）
            if not calib_file:
                # 默认查找标定文件（根据左右手）
                default_calib_paths = [
                    os.path.join(os.path.dirname(__file__), "stereo_calibration", hand_key, f"stereo_calibration_{hand_key}.json"),
                    os.path.join(os.path.dirname(__file__), "..", "stereo_calibration", hand_key, f"stereo_calibration_{hand_key}.json"),
                    f"stereo_calibration/{hand_key}/stereo_calibration_{hand_key}.json",
                    # 兼容旧格式（如果没有左右手目录，查找根目录下的文件）
                    os.path.join(os.path.dirname(__file__), "stereo_calibration", "stereo_calibration.json"),
                    os.path.join(os.path.dirname(__file__), "..", "stereo_calibration", "stereo_calibration.json"),
                    "stereo_calibration/stereo_calibration.json",
                ]
                for path in default_calib_paths:
                    if os.path.exists(path):
                        calib_file = path
                        break
            
            if calib_file and os.path.exists(calib_file):
                try:
                    with open(calib_file, 'r') as f:
                        calib = json.load(f)
                    calib_source = calib_file
                except Exception as e:
                    print(f"[{self.hand_name}] ⚠️ 加载JSON标定文件失败: {e}")
                    calib = None
        
        # 应用标定参数
        if calib:
            try:
                # 转换为numpy数组
                mtx_l = np.array(calib['left_camera_matrix'])
                dist_l = np.array(calib['left_distortion'])
                mtx_r = np.array(calib['right_camera_matrix'])
                dist_r = np.array(calib['right_distortion'])
                R1 = np.array(calib['rectify_rotation_left'])
                R2 = np.array(calib['rectify_rotation_right'])
                P1 = np.array(calib['projection_left'])
                P2 = np.array(calib['projection_right'])
                image_size = tuple(calib['image_size'])
                
                # 计算校正映射
                self.stereo_map1_l, self.stereo_map2_l = cv2.initUndistortRectifyMap(
                    mtx_l, dist_l, R1, P1, image_size, cv2.CV_16SC2
                )
                self.stereo_map1_r, self.stereo_map2_r = cv2.initUndistortRectifyMap(
                    mtx_r, dist_r, R2, P2, image_size, cv2.CV_16SC2
                )
                
                self.stereo_calibration = calib
                self.stereo_rectify_enabled = True
                
                print(f"[{self.hand_name}] ✓ 立体校正已加载: {calib_source}")
                print(f"[{self.hand_name}]   基线距离: {calib.get('baseline_mm', 0):.2f} mm")
                print(f"[{self.hand_name}]   重投影误差: {calib.get('reprojection_error', 0):.3f} 像素")
            except Exception as e:
                print(f"[{self.hand_name}] ⚠️ 应用立体校正失败: {e}")
                print(f"[{self.hand_name}]   将使用原始图像（未校正）")
                self.stereo_rectify_enabled = False
        else:
            print(f"[{self.hand_name}] ⚠️ 未找到立体校正参数，将使用原始图像")
            print(f"[{self.hand_name}]   如需校正，请先运行标定: python stereo_calibration.py --calibrate --hand {hand_key}")
            self.stereo_rectify_enabled = False
    
    def _rectify_stereo(self, stereo_img: np.ndarray) -> np.ndarray:
        """对双目图像应用立体校正"""
        if not self.stereo_rectify_enabled or self.stereo_map1_l is None:
            return stereo_img
        
        # 分割左右图像
        mid = stereo_img.shape[1] // 2
        left_raw = stereo_img[:, :mid]
        right_raw = stereo_img[:, mid:]
        
        # 应用校正
        left_rectified = cv2.remap(left_raw, self.stereo_map1_l, self.stereo_map2_l, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_raw, self.stereo_map1_r, self.stereo_map2_r, cv2.INTER_LINEAR)
        
        # 重新拼接
        rectified = np.hstack([left_rectified, right_rectified])
        return rectified
    
    def warmup_and_calibrate(self, warmup_time: float, calib_time: float):
        """预热并校准"""
        print(f"\n[{self.hand_name}] 预热 {warmup_time}s...")
        
        start = time.time()
        while time.time() - start < warmup_time:
            elapsed = time.time() - start
            s_fps = self.stereo.get_fps()
            m_fps = self.mono.get_fps()
            print(f"\r[{self.hand_name}] [{elapsed:.1f}s] S:{s_fps:.1f}fps M:{m_fps:.1f}fps", end="", flush=True)
            time.sleep(0.1)
        print()
        
        # 清空预热数据
        self.stereo.clear_buffer()
        self.mono.clear_buffer()
        if self.encoder:
            self.encoder.clear_buffer()
        
        # 校准
        print(f"[{self.hand_name}] 校准 {calib_time}s...")
        time.sleep(calib_time)
        
        stereo_data = self.stereo.get_buffer()
        mono_data = self.mono.get_buffer()
        encoder_data = self.encoder.get_buffer() if self.encoder else []
        
        offsets_sm = []
        offsets_se = []
        
        for s in stereo_data:
            if mono_data:
                best = min(mono_data, key=lambda x: abs(x.timestamp - s.timestamp))
                if abs(best.timestamp - s.timestamp) < 0.05:
                    offsets_sm.append((s.timestamp - best.timestamp) * 1000)
            
            if encoder_data:
                best = min(encoder_data, key=lambda x: abs(x.timestamp - s.timestamp))
                if abs(best.timestamp - s.timestamp) < 0.05:
                    offsets_se.append((s.timestamp - best.timestamp) * 1000)
        
        if offsets_sm:
            self.stereo_mono_offset_ms = np.median(offsets_sm)
        if offsets_se:
            self.stereo_encoder_offset_ms = np.median(offsets_se)
        
        print(f"[{self.hand_name}] ✓ 偏移量: S-M={self.stereo_mono_offset_ms:.1f}ms, S-E={self.stereo_encoder_offset_ms:.1f}ms")
        
        # 清空校准数据
        self.stereo.clear_buffer()
        self.mono.clear_buffer()
        if self.encoder:
            self.encoder.clear_buffer()
    
    def start_recording(self) -> bool:
        """开始录制"""
        if not self._ready or self._recording:
            return False
        
        with self._record_lock:
            self._recorded_stereo.clear()
            self._recorded_mono.clear()
            self._recorded_encoder.clear()
            self._record_stats = {'frames': 0, 'last_print': time.time()}
            # 重置增量复制索引
            self._last_copied_idx = {'stereo': 0, 'mono': 0, 'encoder': 0}
        
        self.stereo.clear_buffer()
        self.mono.clear_buffer()
        if self.encoder:
            self.encoder.clear_buffer()
        
        self._record_start_ts = time.time()
        self._recording = True
        
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()
        
        return True
    
    def _record_loop(self):
        """录制循环（增量复制 + 实时JPEG压缩优化）"""
        import os
        
        process = None
        memory_warning_threshold = 3 * 1024 * 1024 * 1024  # 3GB警告阈值（降低以提前预警）
        if HAS_PSUTIL:
            try:
                process = psutil.Process(os.getpid())
            except:
                pass
        
        # JPEG压缩参数
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        
        while self._recording:
            time.sleep(0.1)  # 保持0.1秒采样间隔
            
            # 获取当前缓冲区（全量）
            s_data = self.stereo.get_buffer()
            m_data = self.mono.get_buffer()
            e_data = self.encoder.get_buffer() if self.encoder else []
            
            # 增量复制：只处理新增的帧
            new_stereo = [f for f in s_data if f.idx > self._last_copied_idx['stereo']]
            new_mono = [f for f in m_data if f.idx > self._last_copied_idx['mono']]
            new_encoder = [f for f in e_data if f.idx > self._last_copied_idx['encoder']]
            
            # 实时JPEG压缩新帧
            compressed_stereo = []
            for frame in new_stereo:
                success, jpeg_data = cv2.imencode('.jpg', frame.data, encode_params)
                if success:
                    # 将JPEG字节数据包装为SensorFrame
                    compressed_stereo.append(SensorFrame(frame.timestamp, np.asarray(jpeg_data, dtype=np.uint8), frame.idx))
            
            compressed_mono = []
            for frame in new_mono:
                success, jpeg_data = cv2.imencode('.jpg', frame.data, encode_params)
                if success:
                    compressed_mono.append(SensorFrame(frame.timestamp, np.asarray(jpeg_data, dtype=np.uint8), frame.idx))
            
            # 追加到录制列表（线程安全）
            with self._record_lock:
                self._recorded_stereo.extend(compressed_stereo)
                self._recorded_mono.extend(compressed_mono)
                self._recorded_encoder.extend(new_encoder)
                
                current_frames = len(self._recorded_stereo)
                
                # 更新统计
                if compressed_stereo:
                    self._record_stats['frames'] += len(compressed_stereo)
            
            # 更新索引
            if new_stereo:
                self._last_copied_idx['stereo'] = new_stereo[-1].idx
            if new_mono:
                self._last_copied_idx['mono'] = new_mono[-1].idx
            if new_encoder:
                self._last_copied_idx['encoder'] = new_encoder[-1].idx
            
            # 检查是否达到最大帧数限制
            if current_frames >= self._max_record_frames:
                print(f"\n⚠️ [{self.hand_name}] 达到最大录制帧数 {self._max_record_frames}，自动停止")
                self._recording = False
                break
            
            # 定期打印进度
            now = time.time()
            if now - self._record_stats['last_print'] >= 5.0:
                elapsed = now - self._record_start_ts
                total_fps = self._record_stats['frames'] / elapsed if elapsed > 0 else 0
                
                # 获取双目和单目摄像头的实际帧率
                stereo_fps = self.stereo.get_fps()
                mono_fps = self.mono.get_fps()
                encoder_count = len(self._recorded_encoder)
                
                # 检查内存使用
                if process:
                    try:
                        mem_info = process.memory_info()
                        mem_mb = mem_info.rss / (1024 * 1024)
                        mem_gb = mem_mb / 1024
                        
                        # 内存警告（自动停止）
                        if mem_info.rss > memory_warning_threshold:
                            print(f"\n⚠️ [{self.hand_name}] 内存使用过高: {mem_gb:.1f}GB，自动停止录制")
                            self._recording = False
                            break
                        
                        print(f"[{self.hand_name}] 录制中: {current_frames} 帧 ({elapsed:.1f}s) | 总FPS: {total_fps:.1f} | 双目: {stereo_fps:.1f}fps | 单目: {mono_fps:.1f}fps | encoder: {encoder_count} | 内存: {mem_gb:.1f}GB", end='\r')
                    except:
                        print(f"[{self.hand_name}] 录制中: {current_frames} 帧 ({elapsed:.1f}s) | 总FPS: {total_fps:.1f} | 双目: {stereo_fps:.1f}fps | 单目: {mono_fps:.1f}fps | encoder: {encoder_count}", end='\r')
                else:
                    print(f"[{self.hand_name}] 录制中: {current_frames} 帧 ({elapsed:.1f}s) | 总FPS: {total_fps:.1f} | 双目: {stereo_fps:.1f}fps | 单目: {mono_fps:.1f}fps | encoder: {encoder_count}", end='\r')
                
                self._record_stats['last_print'] = now
            
            # 清空缓冲区（与树莓派版本保持一致）
            self.stereo.clear_buffer()
            self.mono.clear_buffer()
            if self.encoder:
                self.encoder.clear_buffer()
    
    def get_current_frame(self) -> Optional[HandFrame]:
        """获取当前帧（用于实时预览，应用立体校正）"""
        if not self._ready:
            return None
        
        s_data = self.stereo.get_buffer()
        m_data = self.mono.get_buffer()
        e_data = self.encoder.get_buffer() if self.encoder else []
        
        if not s_data or not m_data:
            return None
        
        latest_s = s_data[-1]
        best_m = min(m_data, key=lambda x: abs(x.timestamp - latest_s.timestamp))
        
        angle = 0.0
        enc_ts = latest_s.timestamp
        if e_data:
            best_e = min(e_data, key=lambda x: abs(x.timestamp - latest_s.timestamp))
            angle = best_e.data
            enc_ts = best_e.timestamp
        
        # 应用立体校正
        stereo_rectified = self._rectify_stereo(latest_s.data)
        
        return HandFrame(
            stereo=stereo_rectified,
            mono=best_m.data,
            angle=angle,
            timestamp=latest_s.timestamp,
            stereo_ts=latest_s.timestamp,
            mono_ts=best_m.timestamp,
            encoder_ts=enc_ts,
            idx=0
        )
    
    def stop_recording(self) -> List[HandFrame]:
        """停止录制并返回对齐的数据（分批处理优化内存）"""
        if not self._recording:
            return []
        
        print(f"\n[{self.hand_name}] 停止录制，处理数据...")
        self._recording = False
        if self._record_thread:
            self._record_thread.join(timeout=2)
        
        # 收集最后的数据（增量式）
        s_data = self.stereo.get_buffer()
        m_data = self.mono.get_buffer()
        e_data = self.encoder.get_buffer() if self.encoder else []
        
        # JPEG压缩参数
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        
        # 增量处理最后的新数据
        new_stereo = [f for f in s_data if f.idx > self._last_copied_idx['stereo']]
        new_mono = [f for f in m_data if f.idx > self._last_copied_idx['mono']]
        new_encoder = [f for f in e_data if f.idx > self._last_copied_idx['encoder']]
        
        # 压缩最后的新帧
        compressed_stereo = []
        for frame in new_stereo:
            success, jpeg_data = cv2.imencode('.jpg', frame.data, encode_params)
            if success:
                compressed_stereo.append(SensorFrame(frame.timestamp, np.asarray(jpeg_data, dtype=np.uint8), frame.idx))
        
        compressed_mono = []
        for frame in new_mono:
            success, jpeg_data = cv2.imencode('.jpg', frame.data, encode_params)
            if success:
                compressed_mono.append(SensorFrame(frame.timestamp, np.asarray(jpeg_data, dtype=np.uint8), frame.idx))
        
        with self._record_lock:
            # 添加最后的压缩数据
            self._recorded_stereo.extend(compressed_stereo)
            self._recorded_mono.extend(compressed_mono)
            self._recorded_encoder.extend(new_encoder)
            
            # 复制数据（注意：现在是JPEG压缩的数据）
            stereo_data = list(self._recorded_stereo)
            mono_data = list(self._recorded_mono)
            encoder_data = list(self._recorded_encoder)
        
        # 计算录制统计信息
        record_duration = time.time() - self._record_start_ts
        stereo_fps = len(stereo_data) / record_duration if record_duration > 0 else 0
        mono_fps = len(mono_data) / record_duration if record_duration > 0 else 0
        encoder_fps = len(encoder_data) / record_duration if record_duration > 0 else 0
        
        print(f"[{self.hand_name}] 原始数据: {len(stereo_data)} stereo, {len(mono_data)} mono, {len(encoder_data)} encoder")
        print(f"[{self.hand_name}] 录制时长: {record_duration:.2f}s")
        print(f"[{self.hand_name}] 实际帧率: 双目={stereo_fps:.2f}fps, 单目={mono_fps:.2f}fps, 编码器={encoder_fps:.2f}fps")
        
        # 分批对齐数据（优化内存使用）
        # 使用较宽松的时间差容限（200ms）以适应不同帧率
        aligned = self._align_data_batch(stereo_data, mono_data, encoder_data, max_time_diff_ms=200.0)
        
        # 清理原始数据，释放内存
        del stereo_data, mono_data, encoder_data
        
        print(f"[{self.hand_name}] 对齐后: {len(aligned)} 帧")
        
        return aligned
    
    def _align_data(self, stereo_data: List[SensorFrame], 
                   mono_data: List[SensorFrame],
                   encoder_data: List[SensorFrame],
                   max_time_diff_ms: float = 50.0) -> List[HandFrame]:
        """对齐数据（应用立体校正）- 旧版本，保留兼容性"""
        return self._align_data_batch(stereo_data, mono_data, encoder_data, max_time_diff_ms)
    
    def _align_data_batch(self, stereo_data: List[SensorFrame], 
                          mono_data: List[SensorFrame],
                          encoder_data: List[SensorFrame],
                          max_time_diff_ms: float = 50.0,
                          batch_size: int = 100) -> List[HandFrame]:
        """分批对齐数据（优化内存使用，处理JPEG压缩数据）"""
        aligned = []
        n_total = len(stereo_data)
        
        # 如果数据量不大，直接处理
        if n_total <= batch_size:
            for s in stereo_data:
                mono_target = s.timestamp - self.stereo_mono_offset_ms / 1000.0
                encoder_target = s.timestamp - self.stereo_encoder_offset_ms / 1000.0
                
                mono = None
                if mono_data:
                    best = min(mono_data, key=lambda x: abs(x.timestamp - mono_target))
                    if abs(best.timestamp - mono_target) * 1000 <= max_time_diff_ms:
                        mono = best
                
                enc = None
                if encoder_data:
                    best = min(encoder_data, key=lambda x: abs(x.timestamp - encoder_target))
                    if abs(best.timestamp - encoder_target) * 1000 <= max_time_diff_ms:
                        enc = best
                
                # 允许没有编码器数据的情况（使用默认角度0）
                if mono:
                    # 解压JPEG数据
                    stereo_img = cv2.imdecode(s.data, cv2.IMREAD_COLOR)
                    mono_img = cv2.imdecode(mono.data, cv2.IMREAD_COLOR)
                    
                    # 应用立体校正
                    stereo_rectified = self._rectify_stereo(stereo_img)
                    
                    aligned.append(HandFrame(
                        stereo=stereo_rectified,
                        mono=mono_img,
                        angle=enc.data if enc else 0.0,
                        timestamp=s.timestamp,
                        stereo_ts=s.timestamp,
                        mono_ts=mono.timestamp,
                        encoder_ts=enc.timestamp if enc else s.timestamp,
                        idx=len(aligned) + 1
                    ))
        else:
            # 分批处理
            for i in range(0, n_total, batch_size):
                batch_end = min(i + batch_size, n_total)
                batch_stereo = stereo_data[i:batch_end]
                
                batch_aligned = []
                for s in batch_stereo:
                    mono_target = s.timestamp - self.stereo_mono_offset_ms / 1000.0
                    encoder_target = s.timestamp - self.stereo_encoder_offset_ms / 1000.0
                    
                    mono = None
                    if mono_data:
                        best = min(mono_data, key=lambda x: abs(x.timestamp - mono_target))
                        if abs(best.timestamp - mono_target) * 1000 <= max_time_diff_ms:
                            mono = best
                    
                    enc = None
                    if encoder_data:
                        best = min(encoder_data, key=lambda x: abs(x.timestamp - encoder_target))
                        if abs(best.timestamp - encoder_target) * 1000 <= max_time_diff_ms:
                            enc = best
                    
                    # 允许没有编码器数据的情况（使用默认角度0）
                    if mono:
                        # 解压JPEG数据
                        stereo_img = cv2.imdecode(s.data, cv2.IMREAD_COLOR)
                        mono_img = cv2.imdecode(mono.data, cv2.IMREAD_COLOR)
                        
                        # 应用立体校正
                        stereo_rectified = self._rectify_stereo(stereo_img)
                        
                        batch_aligned.append(HandFrame(
                            stereo=stereo_rectified,
                            mono=mono_img,
                            angle=enc.data if enc else 0.0,
                            timestamp=s.timestamp,
                            stereo_ts=s.timestamp,
                            mono_ts=mono.timestamp,
                            encoder_ts=enc.timestamp if enc else s.timestamp,
                        idx=len(aligned) + len(batch_aligned) + 1
                    ))
                
                aligned.extend(batch_aligned)
                
                # 显示进度
                progress = (batch_end / n_total) * 100
                print(f"[{self.hand_name}] 对齐进度: {batch_end}/{n_total} ({progress:.1f}%)", end='\r')
            
            print()  # 换行
        
        return aligned
    
    def stop(self):
        """停止"""
        self._recording = False
        self._running = False
        self._ready = False
        
        if self._record_thread:
            self._record_thread.join(timeout=2)
        
        if self.stereo:
            self.stereo.stop()
        if self.mono:
            self.mono.stop()
        if self.encoder:
            self.encoder.stop()
        
        print(f"[{self.hand_name}] 已停止")


class DualHandCollector:
    """双手同步数据收集器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
        self.left: Optional[HandCollector] = None
        self.right: Optional[HandCollector] = None
        
        self._ready = False
        self._running = False
        self._recording = False
    
    def _load_config(self) -> dict:
        """加载配置"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @property
    def is_ready(self) -> bool:
        return self._ready
    
    @property
    def is_recording(self) -> bool:
        return self._recording
    
    def start(self) -> bool:
        """启动收集器"""
        if self._running:
            return True
        
        self._running = True
        self._ready = False
        
        # 初始化左右手收集器
        left_cfg = self.config.get('left_hand', {})
        right_cfg = self.config.get('right_hand', {})
        
        self.left = HandCollector(left_cfg, "LEFT")
        self.right = HandCollector(right_cfg, "RIGHT")
        
        # 启动
        self.left.start()
        self.right.start()
        
        # 等待就绪
        if not self.left.wait_ready() or not self.right.wait_ready():
            print("❌ 初始化失败")
            return False
        
        # 预热和校准
        sync_cfg = self.config.get('sync', {})
        warmup_time = sync_cfg.get('warmup_time', 1.0)
        calib_time = sync_cfg.get('calib_time', 0.5)
        
        self.left.warmup_and_calibrate(warmup_time, calib_time)
        self.right.warmup_and_calibrate(warmup_time, calib_time)
        
        self._ready = True
        print("\n✅ 双手收集器就绪")
        return True
    
    def wait_ready(self, timeout: float = 60.0) -> bool:
        """等待就绪"""
        start = time.time()
        while not self._ready and (time.time() - start) < timeout:
            time.sleep(0.1)
        return self._ready
    
    def start_recording(self) -> bool:
        """开始录制"""
        if not self._ready or self._recording:
            return False
        
        self._recording = True
        self.left.start_recording()
        self.right.start_recording()
        
        print("🔴 开始录制双手数据...")
        return True
    
    def stop_recording(self) -> List[DualHandFrame]:
        """停止录制并返回对齐的双手数据"""
        if not self._recording:
            return []
        
        self._recording = False
        
        # 获取左右手数据
        left_data = self.left.stop_recording()
        right_data = self.right.stop_recording()
        
        # 对齐双手数据
        aligned = self._align_hands(left_data, right_data)
        
        print(f"\n双手对齐完成: {len(aligned)} 帧")
        print(f"  左手: {len(left_data)} 帧")
        print(f"  右手: {len(right_data)} 帧")
        
        return aligned
    
    def _align_hands(self, left_data: List[HandFrame], 
                    right_data: List[HandFrame],
                    max_time_diff_ms: float = 50.0) -> List[DualHandFrame]:
        """对齐左右手数据"""
        aligned = []
        
        # 以左手数据为基准对齐
        for left_frame in left_data:
            target_ts = left_frame.timestamp
            
            # 找到最接近的右手帧
            best_right = None
            if right_data:
                best_right = min(right_data, key=lambda x: abs(x.timestamp - target_ts))
                if abs(best_right.timestamp - target_ts) * 1000 > max_time_diff_ms:
                    best_right = None
            
            if best_right:
                aligned.append(DualHandFrame(
                    left=left_frame,
                    right=best_right,
                    timestamp=target_ts,
                    idx=len(aligned) + 1
                ))
        
        return aligned
    
    def stop(self):
        """停止"""
        self._recording = False
        self._running = False
        self._ready = False
        
        if self.left:
            self.left.stop()
        if self.right:
            self.right.stop()
        
        print("双手收集器已停止")


def save_dual_hand_data(data: List[DualHandFrame], output_dir: str,
                        jpeg_quality: int = 85) -> str:
    """保存双手数据到HDF5（流式写入优化内存）"""
    if not data:
        print("❌ 无数据")
        return None
    
    if not HAS_H5PY:
        print("❌ h5py 未安装，无法保存数据")
        return None
    
    from datetime import datetime
    import h5py
    
    os.makedirs(output_dir, exist_ok=True)
    prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_dual_hand_data.h5"
    filepath = os.path.join(output_dir, filename)
    
    n_frames = len(data)
    print(f"\n保存双手数据: {filepath}")
    print(f"  帧数: {n_frames}")
    
    start_time = time.time()
    
    # 流式写入HDF5（避免一次性加载所有数据到内存）
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    
    print("  写入HDF5（流式）...")
    write_start = time.time()
    
    with h5py.File(filepath, 'w', libver='latest') as f:
        # 元数据
        f.attrs['n_frames'] = n_frames
        f.attrs['left_stereo_shape'] = data[0].left.stereo.shape
        f.attrs['left_mono_shape'] = data[0].left.mono.shape
        f.attrs['right_stereo_shape'] = data[0].right.stereo.shape
        f.attrs['right_mono_shape'] = data[0].right.mono.shape
        f.attrs['jpeg_quality'] = jpeg_quality
        f.attrs['created_at'] = datetime.now().isoformat()
        
        # 创建数据集（使用可变长度数据类型）
        dt = h5py.special_dtype(vlen=np.uint8)
        
        # 创建数据集（预先分配空间）
        left_stereo_ds = f.create_dataset('left_stereo_jpeg', (n_frames,), dtype=dt)
        left_mono_ds = f.create_dataset('left_mono_jpeg', (n_frames,), dtype=dt)
        right_stereo_ds = f.create_dataset('right_stereo_jpeg', (n_frames,), dtype=dt)
        right_mono_ds = f.create_dataset('right_mono_jpeg', (n_frames,), dtype=dt)
        
        left_angles = np.zeros(n_frames, dtype=np.float32)
        left_timestamps = np.zeros(n_frames, dtype=np.float64)
        right_angles = np.zeros(n_frames, dtype=np.float32)
        right_timestamps = np.zeros(n_frames, dtype=np.float64)
        sync_timestamps = np.zeros(n_frames, dtype=np.float64)
        
        # 分批处理和写入（每批100帧）
        batch_size = 100
        for i in range(0, n_frames, batch_size):
            batch_end = min(i + batch_size, n_frames)
            batch_data = data[i:batch_end]
            
            for j, frame in enumerate(batch_data):
                idx = i + j
                
                # 压缩图像（立即写入，不保存到列表）
                success_ls, ls = cv2.imencode('.jpg', frame.left.stereo, encode_params)
                success_lm, lm = cv2.imencode('.jpg', frame.left.mono, encode_params)
                success_rs, rs = cv2.imencode('.jpg', frame.right.stereo, encode_params)
                success_rm, rm = cv2.imencode('.jpg', frame.right.mono, encode_params)
                
                if not (success_ls and success_lm and success_rs and success_rm):
                    print(f"\n⚠️ 警告: 第 {idx} 帧图像压缩失败，跳过")
                    continue
                
                # 对于可变长度数据类型，直接传递numpy数组（h5py会自动处理）
                left_stereo_ds[idx] = ls
                left_mono_ds[idx] = lm
                right_stereo_ds[idx] = rs
                right_mono_ds[idx] = rm
                
                left_angles[idx] = frame.left.angle
                left_timestamps[idx] = frame.left.timestamp
                right_angles[idx] = frame.right.angle
                right_timestamps[idx] = frame.right.timestamp
                sync_timestamps[idx] = frame.timestamp
            
            # 显示进度
            progress = (batch_end / n_frames) * 100
            print(f"  进度: {batch_end}/{n_frames} ({progress:.1f}%)", end='\r')
        
        print()  # 换行
        
        # 写入角度和时间戳数据
        f.create_dataset('left_angles', data=left_angles, dtype=np.float32)
        f.create_dataset('left_timestamps', data=left_timestamps, dtype=np.float64)
        f.create_dataset('right_angles', data=right_angles, dtype=np.float32)
        f.create_dataset('right_timestamps', data=right_timestamps, dtype=np.float64)
        f.create_dataset('sync_timestamps', data=sync_timestamps, dtype=np.float64)
    
    write_time = time.time() - write_start
    total_time = time.time() - start_time
    
    file_size = os.path.getsize(filepath) / (1024 * 1024)
    
    print(f"  写入耗时: {write_time:.2f}s")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  文件大小: {file_size:.1f}MB")
    print(f"✅ 保存完成: {filepath}")
    
    return filepath


def visualize_hand(collector: HandCollector, hand_name: str):
    """可视化单手数据"""
    print(f"\n[{hand_name}] 开始可视化（按 'q' 退出）...")
    
    window_name = f"{hand_name} Hand - Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    angle_history = []
    max_history = 100
    
    try:
        while True:
            frame = collector.get_current_frame()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # 获取图像
            stereo_img = frame.stereo.copy()
            mono_img = frame.mono.copy()
            angle = frame.angle
            
            # 更新角度历史
            angle_history.append(angle)
            if len(angle_history) > max_history:
                angle_history.pop(0)
            
            # 分割双目图像（假设是左右拼接的）
            stereo_h, stereo_w = stereo_img.shape[:2]
            if stereo_w > stereo_h:
                # 水平拼接：左右
                left_img = stereo_img[:, :stereo_w//2]
                right_img = stereo_img[:, stereo_w//2:]
            else:
                # 垂直拼接：上下
                left_img = stereo_img[:stereo_h//2, :]
                right_img = stereo_img[stereo_h//2:, :]
            
            # 保持原始尺寸，不进行resize
            mono_display = mono_img.copy()
            left_display = left_img.copy()
            right_display = right_img.copy()
            
            # 创建角度显示图像（高度与mono_display匹配）
            angle_img = np.zeros((mono_display.shape[0], 400, 3), dtype=np.uint8)
            
            # 绘制当前角度
            angle_text = f"Angle: {angle:.2f}°"
            cv2.putText(angle_img, angle_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 绘制角度历史曲线
            if len(angle_history) > 1:
                points = []
                angle_img_h = angle_img.shape[0]
                angle_img_w = angle_img.shape[1]
                for i, a in enumerate(angle_history):
                    x = int((i / max(len(angle_history) - 1, 1)) * (angle_img_w - 20) + 10)
                    # 角度范围假设是-180到180，映射到图像高度
                    y = int(angle_img_h - 20 - (a + 180) / 360 * (angle_img_h - 40))
                    points.append((x, y))
                
                for i in range(len(points) - 1):
                    cv2.line(angle_img, points[i], points[i+1], (0, 255, 255), 2)
            
            # 组合显示图像
            # 上排：双目左右
            stereo_row = np.hstack([left_display, right_display])
            
            # 下排：单目和角度
            bottom_row = np.hstack([mono_display, angle_img])
            
            # 如果宽度不一致，用黑色填充较小的图像
            if stereo_row.shape[1] != bottom_row.shape[1]:
                target_width = max(stereo_row.shape[1], bottom_row.shape[1])
                if stereo_row.shape[1] < target_width:
                    # 在右侧填充黑色
                    padding = np.zeros((stereo_row.shape[0], target_width - stereo_row.shape[1], 3), dtype=np.uint8)
                    stereo_row = np.hstack([stereo_row, padding])
                if bottom_row.shape[1] < target_width:
                    # 在右侧填充黑色
                    padding = np.zeros((bottom_row.shape[0], target_width - bottom_row.shape[1], 3), dtype=np.uint8)
                    bottom_row = np.hstack([bottom_row, padding])
            
            display = np.vstack([stereo_row, bottom_row])
            
            # 添加标题
            title_height = 30
            title_img = np.zeros((title_height, display.shape[1], 3), dtype=np.uint8)
            title_text = f"{hand_name} Hand - Stereo (Left/Right) | Mono | Angle"
            cv2.putText(title_img, title_text, (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            display = np.vstack([title_img, display])
            
            # 显示FPS和时间戳
            fps_text = f"FPS: {1.0 / (time.time() - getattr(visualize_hand, 'last_time', time.time())):.1f}"
            cv2.putText(display, fps_text, (10, display.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            visualize_hand.last_time = time.time()
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyWindow(window_name)


def visualize_dual_hand(collector: DualHandCollector):
    """可视化双手数据"""
    print("\n双手可视化（按 'q' 退出）...")
    
    window_name = "Dual Hand - Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    left_angle_history = []
    right_angle_history = []
    max_history = 100
    
    try:
        while True:
            left_frame = collector.left.get_current_frame() if collector.left else None
            right_frame = collector.right.get_current_frame() if collector.right else None
            
            if left_frame is None and right_frame is None:
                time.sleep(0.01)
                continue
            
            displays = []
            
            # 处理左手
            if left_frame:
                stereo_img = left_frame.stereo.copy()
                mono_img = left_frame.mono.copy()
                angle = left_frame.angle
                
                left_angle_history.append(angle)
                if len(left_angle_history) > max_history:
                    left_angle_history.pop(0)
                
                # 分割双目
                stereo_h, stereo_w = stereo_img.shape[:2]
                if stereo_w > stereo_h:
                    left_stereo = stereo_img[:, :stereo_w//2]
                    right_stereo = stereo_img[:, stereo_w//2:]
                else:
                    left_stereo = stereo_img[:stereo_h//2, :]
                    right_stereo = stereo_img[stereo_h//2:, :]
                
                # 保持原始尺寸，不进行resize
                mono_display = mono_img.copy()
                left_display = left_stereo.copy()
                right_display = right_stereo.copy()
                
                # 角度显示（高度与mono_display匹配）
                angle_img = np.zeros((mono_display.shape[0], 300, 3), dtype=np.uint8)
                angle_img_h = angle_img.shape[0]
                angle_img_w = angle_img.shape[1]
                cv2.putText(angle_img, f"L: {angle:.2f}°", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if len(left_angle_history) > 1:
                    points = []
                    for i, a in enumerate(left_angle_history):
                        x = int((i / max(len(left_angle_history) - 1, 1)) * (angle_img_w - 20) + 10)
                        y = int(angle_img_h - 20 - (a + 180) / 360 * (angle_img_h - 40))
                        points.append((x, y))
                    for i in range(len(points) - 1):
                        cv2.line(angle_img, points[i], points[i+1], (0, 255, 255), 2)
                
                left_row = np.hstack([left_display, right_display])
                left_bottom = np.hstack([mono_display, angle_img])
                if left_row.shape[1] != left_bottom.shape[1]:
                    target_w = max(left_row.shape[1], left_bottom.shape[1])
                    if left_row.shape[1] < target_w:
                        # 在右侧填充黑色
                        padding = np.zeros((left_row.shape[0], target_w - left_row.shape[1], 3), dtype=np.uint8)
                        left_row = np.hstack([left_row, padding])
                    if left_bottom.shape[1] < target_w:
                        # 在右侧填充黑色
                        padding = np.zeros((left_bottom.shape[0], target_w - left_bottom.shape[1], 3), dtype=np.uint8)
                        left_bottom = np.hstack([left_bottom, padding])
                
                left_display_full = np.vstack([left_row, left_bottom])
                displays.append(left_display_full)
            
            # 处理右手
            if right_frame:
                stereo_img = right_frame.stereo.copy()
                mono_img = right_frame.mono.copy()
                angle = right_frame.angle
                
                right_angle_history.append(angle)
                if len(right_angle_history) > max_history:
                    right_angle_history.pop(0)
                
                # 分割双目
                stereo_h, stereo_w = stereo_img.shape[:2]
                if stereo_w > stereo_h:
                    left_stereo = stereo_img[:, :stereo_w//2]
                    right_stereo = stereo_img[:, stereo_w//2:]
                else:
                    left_stereo = stereo_img[:stereo_h//2, :]
                    right_stereo = stereo_img[stereo_h//2:, :]
                
                # 保持原始尺寸，不进行resize
                mono_display = mono_img.copy()
                left_display = left_stereo.copy()
                right_display = right_stereo.copy()
                
                # 角度显示（高度与mono_display匹配）
                angle_img = np.zeros((mono_display.shape[0], 300, 3), dtype=np.uint8)
                angle_img_h = angle_img.shape[0]
                angle_img_w = angle_img.shape[1]
                cv2.putText(angle_img, f"R: {angle:.2f}°", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if len(right_angle_history) > 1:
                    points = []
                    for i, a in enumerate(right_angle_history):
                        x = int((i / max(len(right_angle_history) - 1, 1)) * (angle_img_w - 20) + 10)
                        y = int(angle_img_h - 20 - (a + 180) / 360 * (angle_img_h - 40))
                        points.append((x, y))
                    for i in range(len(points) - 1):
                        cv2.line(angle_img, points[i], points[i+1], (0, 255, 255), 2)
                
                right_row = np.hstack([left_display, right_display])
                right_bottom = np.hstack([mono_display, angle_img])
                if right_row.shape[1] != right_bottom.shape[1]:
                    target_w = max(right_row.shape[1], right_bottom.shape[1])
                    if right_row.shape[1] < target_w:
                        # 在右侧填充黑色
                        padding = np.zeros((right_row.shape[0], target_w - right_row.shape[1], 3), dtype=np.uint8)
                        right_row = np.hstack([right_row, padding])
                    if right_bottom.shape[1] < target_w:
                        # 在右侧填充黑色
                        padding = np.zeros((right_bottom.shape[0], target_w - right_bottom.shape[1], 3), dtype=np.uint8)
                        right_bottom = np.hstack([right_bottom, padding])
                
                right_display_full = np.vstack([right_row, right_bottom])
                displays.append(right_display_full)
            
            if displays:
                # 组合显示
                if len(displays) == 2:
                    # 双手：上下排列
                    final_display = np.vstack(displays)
                else:
                    final_display = displays[0]
                
                # 添加标题
                title_height = 30
                title_img = np.zeros((title_height, final_display.shape[1], 3), dtype=np.uint8)
                title_text = "Dual Hand Visualization - Press 'q' to quit"
                cv2.putText(title_img, title_text, (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                final_display = np.vstack([title_img, final_display])
                
                cv2.imshow(window_name, final_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="双手数据收集器")
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                       help="配置文件路径")
    parser.add_argument("--mode", "-m", type=str, choices=['left', 'right', 'both'],
                       default='both', help="测试模式: left(左手), right(右手), both(双手)")
    parser.add_argument("--visualize", "-v", action="store_true",
                       help="启用可视化模式")
    parser.add_argument("--record", "-r", action="store_true",
                       help="录制模式（与可视化模式互斥）")
    args = parser.parse_args()
    
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    
    # 如果没有指定模式，默认使用可视化
    if not args.visualize and not args.record:
        args.visualize = True
    
    try:
        if args.mode == 'both':
            # 双手模式
            collector = DualHandCollector(config_path)
            collector.start()
            if not collector.wait_ready():
                print("❌ 初始化失败")
                sys.exit(1)
            
            if args.visualize:
                visualize_dual_hand(collector)
            elif args.record:
                print("\n按回车键开始录制...")
                input()
                collector.start_recording()
                print("录制中... 按回车键停止录制")
                input()
                data = collector.stop_recording()
                if data:
                    save_cfg = collector.config.get('save', {})
                    output_dir = save_cfg.get('output_dir', './data')
                    jpeg_quality = save_cfg.get('jpeg_quality', 85)
                    save_dual_hand_data(data, output_dir, jpeg_quality)
            collector.stop()
        
        else:
            # 单手模式
            config = yaml.safe_load(open(config_path, 'r'))
            hand_name = args.mode.upper()
            hand_config = config.get(f'{args.mode}_hand', {})
            
            if not hand_config:
                print(f"❌ 配置文件中没有找到 {args.mode}_hand 配置")
                sys.exit(1)
            
            collector = HandCollector(hand_config, hand_name)
            collector.start()
            if not collector.wait_ready():
                print("❌ 初始化失败")
                sys.exit(1)
            
            # 预热和校准
            sync_cfg = config.get('sync', {})
            warmup_time = sync_cfg.get('warmup_time', 1.0)
            calib_time = sync_cfg.get('calib_time', 0.5)
            collector.warmup_and_calibrate(warmup_time, calib_time)
            
            if args.visualize:
                visualize_hand(collector, hand_name)
            elif args.record:
                print("\n按回车键开始录制...")
                input()
                collector.start_recording()
                print("录制中... 按回车键停止录制")
                input()
                data = collector.stop_recording()
                if data:
                    print(f"\n录制完成: {len(data)} 帧")
                    # 单手数据可以保存为单独的HDF5文件
                    save_cfg = config.get('save', {})
                    output_dir = save_cfg.get('output_dir', './data')
                    jpeg_quality = save_cfg.get('jpeg_quality', 85)
                    # 这里可以添加单手保存函数，暂时只打印
                    print(f"数据已收集，可扩展保存功能")
            
            collector.stop()
        
    except KeyboardInterrupt:
        print("\n中断")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n错误: {e}")


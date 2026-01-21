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
from typing import Optional, List, Dict, Tuple
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
        
        # 录制数据
        self._record_start_ts = 0
        self._recorded_stereo: List[SensorFrame] = []
        self._recorded_mono: List[SensorFrame] = []
        self._recorded_encoder: List[SensorFrame] = []
        self._record_lock = threading.Lock()
        self._record_thread = None
        self._record_stats = {'frames': 0, 'last_print': 0}
    
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
            self._recorded_stereo = []
            self._recorded_mono = []
            self._recorded_encoder = []
            self._record_stats = {'frames': 0, 'last_print': time.time()}
        
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
        """录制循环（参考树莓派版本的实现）"""
        import os
        
        process = None
        memory_warning_threshold = 4 * 1024 * 1024 * 1024  # 4GB警告阈值
        if HAS_PSUTIL:
            try:
                process = psutil.Process(os.getpid())
            except:
                pass
        
        while self._recording:
            time.sleep(0.1)  # 与树莓派版本保持一致：0.1秒
            
            s_data = self.stereo.get_buffer()
            m_data = self.mono.get_buffer()
            e_data = self.encoder.get_buffer() if self.encoder else []
            
            # 直接记录所有数据，不添加条件判断（参考树莓派版本）
            with self._record_lock:
                self._recorded_stereo.extend(s_data)
                self._recorded_mono.extend(m_data)
                self._recorded_encoder.extend(e_data)
                
                # 更新统计信息
                if s_data:
                    self._record_stats['frames'] += len(s_data)
                
                # 每5秒打印一次进度
                now = time.time()
                if now - self._record_stats['last_print'] >= 5.0:
                    elapsed = now - self._record_start_ts
                    fps = self._record_stats['frames'] / elapsed if elapsed > 0 else 0
                    
                    # 检查内存使用
                    if process:
                        try:
                            mem_info = process.memory_info()
                            mem_mb = mem_info.rss / (1024 * 1024)
                            mem_gb = mem_mb / 1024
                            
                            if mem_info.rss > memory_warning_threshold:
                                print(f"\n[{self.hand_name}] ⚠️ 内存使用较高: {mem_gb:.1f}GB ({mem_mb:.0f}MB)")
                            
                            encoder_count = len(self._recorded_encoder)
                            print(f"[{self.hand_name}] 录制中: {self._record_stats['frames']} 帧 ({elapsed:.1f}s, {fps:.1f} fps, {encoder_count} encoder, {mem_gb:.1f}GB)", end='\r')
                        except:
                            encoder_count = len(self._recorded_encoder)
                            print(f"[{self.hand_name}] 录制中: {self._record_stats['frames']} 帧 ({elapsed:.1f}s, {fps:.1f} fps, {encoder_count} encoder)", end='\r')
                    else:
                        encoder_count = len(self._recorded_encoder)
                        print(f"[{self.hand_name}] 录制中: {self._record_stats['frames']} 帧 ({elapsed:.1f}s, {fps:.1f} fps, {encoder_count} encoder)", end='\r')
                    
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
        
        # 收集最后的数据（参考树莓派版本的实现）
        s_data = self.stereo.get_buffer()
        m_data = self.mono.get_buffer()
        e_data = self.encoder.get_buffer() if self.encoder else []
        
        with self._record_lock:
            # 添加最后的数据
            self._recorded_stereo.extend(s_data)
            self._recorded_mono.extend(m_data)
            self._recorded_encoder.extend(e_data)
            
            # 复制数据
            stereo_data = list(self._recorded_stereo)
            mono_data = list(self._recorded_mono)
            encoder_data = list(self._recorded_encoder)
        
        print(f"[{self.hand_name}] 原始数据: {len(stereo_data)} stereo, {len(mono_data)} mono, {len(encoder_data)} encoder")
        
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
        """分批对齐数据（优化内存使用）"""
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
                    # 应用立体校正
                    stereo_rectified = self._rectify_stereo(s.data)
                    
                    aligned.append(HandFrame(
                        stereo=stereo_rectified,
                        mono=mono.data,
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
                        # 应用立体校正
                        stereo_rectified = self._rectify_stereo(s.data)
                        
                    batch_aligned.append(HandFrame(
                        stereo=stereo_rectified,
                        mono=mono.data,
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



def visualize_hand(collector: HandCollector, hand_name: str):
    """可视化单手数据"""
    print(f"\n[{hand_name}] 开始可视化（按 'q' 退出）...")
    
    window_name = f"{hand_name} Hand - Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    angle_history = []
    max_history = 100
    
    # 内存监控
    process = None
    initial_memory_mb = 0
    if HAS_PSUTIL:
        try:
            import psutil
            process = psutil.Process(os.getpid())
            initial_memory_mb = process.memory_info().rss / (1024 * 1024)
        except:
            pass
    
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
            
            # 获取内存信息
            mem_text = ""
            if process:
                try:
                    mem_mb = process.memory_info().rss / (1024 * 1024)
                    mem_delta = mem_mb - initial_memory_mb
                    mem_text = f"Memory: {mem_mb:.0f}MB (+{mem_delta:.0f}MB)"
                except:
                    pass
            
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
            angle_text = f"Angle: {angle:.2f}"
            cv2.putText(angle_img, angle_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 绘制内存信息
            if mem_text:
                cv2.putText(angle_img, mem_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
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
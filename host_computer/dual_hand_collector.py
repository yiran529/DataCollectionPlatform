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
from typing import Optional, List, Tuple, Union
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


from hand_collector import HandCollector, HandFrame, visualize_hand

@dataclass
class DualHandFrame:
    """双手同步帧"""
    left: HandFrame
    right: HandFrame
    timestamp: float
    idx: int

class DualHandCollector:
    """双手同步数据收集器"""
    
    def __init__(self, config_path: str, *, enable_realtime_write: bool = True,
                 output_dir: Optional[str] = None, jpeg_quality: Optional[int] = None):
        self.config_path = config_path
        self.config = self._load_config()
        
        save_cfg = self.config.get('save', {})
        self.enable_realtime_write = enable_realtime_write
        self.output_dir = output_dir if output_dir is not None else save_cfg.get('output_dir', './data')
        self.jpeg_quality = jpeg_quality if jpeg_quality is not None else save_cfg.get('jpeg_quality', 85)
        
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
        
        self.left = HandCollector(
            left_cfg,
            "LEFT",
            enable_realtime_write=self.enable_realtime_write,
            output_dir=self.output_dir,
            jpeg_quality=self.jpeg_quality
        )
        self.right = HandCollector(
            right_cfg,
            "RIGHT",
            enable_realtime_write=self.enable_realtime_write,
            output_dir=self.output_dir,
            jpeg_quality=self.jpeg_quality
        )
        
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
    
    def stop_recording(self) -> Union[Tuple[Optional[str], Optional[str]], List[DualHandFrame]]:
        """停止录制并返回对齐的双手数据或文件路径"""
        if not self._recording:
            return (None, None) if self.enable_realtime_write else []
        
        self._recording = False
        
        # 获取左右手结果
        left_result = self.left.stop_recording()
        right_result = self.right.stop_recording()
        
        if self.enable_realtime_write:
            left_path = left_result if isinstance(left_result, str) else None
            right_path = right_result if isinstance(right_result, str) else None
            if left_path and right_path:
                print(f"\n双手实时写入完成")
                print(f"  左手文件: {left_path}")
                print(f"  右手文件: {right_path}")
            else:
                print("\n⚠️ 实时写入结果缺失，请检查左右手文件是否生成成功")
            return (left_path, right_path)
        
        left_data = left_result if isinstance(left_result, list) else []
        right_data = right_result if isinstance(right_result, list) else []
        
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


def merge_dual_hand_files(left_path: Optional[str], right_path: Optional[str],
                          output_dir: Optional[str] = None,
                          max_time_diff_ms: float = 50.0) -> Optional[str]:
    """离线对齐左右手实时写入文件并合并为一个双手数据文件"""
    if not HAS_H5PY:
        print("❌ h5py 未安装，无法合并双手数据")
        return None
    if not left_path or not right_path:
        print("❌ 缺少左右手文件路径，无法合并")
        return None
    if not os.path.exists(left_path):
        print(f"❌ 左手文件不存在: {left_path}")
        return None
    if not os.path.exists(right_path):
        print(f"❌ 右手文件不存在: {right_path}")
        return None

    output_dir = output_dir or os.path.dirname(left_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        from datetime import datetime
        max_diff_s = max_time_diff_ms / 1000.0
        with h5py.File(left_path, 'r') as left_f, h5py.File(right_path, 'r') as right_f:
            left_ts = np.asarray(left_f['timestamps'], dtype=np.float64)
            right_ts = np.asarray(right_f['timestamps'], dtype=np.float64)
            n_left = left_ts.shape[0]
            n_right = right_ts.shape[0]
            if n_left == 0 or n_right == 0:
                print("❌ 左右手文件中存在空数据集，无法合并")
                return None

            pairs: List[Tuple[int, int]] = []
            r_idx = 0
            while r_idx < n_right and right_ts[r_idx] < left_ts[0] - max_diff_s:
                r_idx += 1
            for l_idx, l_ts in enumerate(left_ts):
                while r_idx < n_right and right_ts[r_idx] < l_ts - max_diff_s:
                    r_idx += 1
                if r_idx >= n_right:
                    break
                best_idx = r_idx
                while (best_idx + 1 < n_right and
                       abs(right_ts[best_idx + 1] - l_ts) <= abs(right_ts[best_idx] - l_ts)):
                    best_idx += 1
                if abs(right_ts[best_idx] - l_ts) <= max_diff_s:
                    pairs.append((l_idx, best_idx))
                    if best_idx + 1 < n_right:
                        r_idx = best_idx + 1
                    else:
                        r_idx = n_right

            if not pairs:
                print("❌ 未找到可对齐的左右手帧，请检查数据时间戳")
                return None

            n_aligned = len(pairs)
            prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_dual_hand_data.h5"
            merged_path = os.path.join(output_dir, filename)

            with h5py.File(merged_path, 'w', libver='latest') as out_f:
                out_f.attrs['n_frames'] = n_aligned
                out_f.attrs['jpeg_quality'] = left_f.attrs.get('jpeg_quality', 85)
                out_f.attrs['created_at'] = datetime.now().isoformat()
                out_f.attrs['left_source'] = os.path.basename(left_path)
                out_f.attrs['right_source'] = os.path.basename(right_path)
                out_f.attrs['merge_mode'] = 'offline_align'
                out_f.attrs['max_time_diff_ms'] = max_time_diff_ms
                if 'hand' in left_f.attrs:
                    out_f.attrs['left_hand'] = left_f.attrs['hand']
                if 'hand' in right_f.attrs:
                    out_f.attrs['right_hand'] = right_f.attrs['hand']
                if 'stereo_shape' in left_f.attrs:
                    out_f.attrs['left_stereo_shape'] = left_f.attrs['stereo_shape']
                if 'mono_shape' in left_f.attrs:
                    out_f.attrs['left_mono_shape'] = left_f.attrs['mono_shape']
                if 'stereo_shape' in right_f.attrs:
                    out_f.attrs['right_stereo_shape'] = right_f.attrs['stereo_shape']
                if 'mono_shape' in right_f.attrs:
                    out_f.attrs['right_mono_shape'] = right_f.attrs['mono_shape']

                dt = h5py.special_dtype(vlen=np.uint8)
                left_stereo_ds = out_f.create_dataset('left_stereo_jpeg', (n_aligned,), dtype=dt)
                left_mono_ds = out_f.create_dataset('left_mono_jpeg', (n_aligned,), dtype=dt)
                right_stereo_ds = out_f.create_dataset('right_stereo_jpeg', (n_aligned,), dtype=dt)
                right_mono_ds = out_f.create_dataset('right_mono_jpeg', (n_aligned,), dtype=dt)

                left_angles = np.zeros(n_aligned, dtype=np.float32)
                left_timestamps = np.zeros(n_aligned, dtype=np.float64)
                left_stereo_ts = np.zeros(n_aligned, dtype=np.float64)
                left_mono_ts = np.zeros(n_aligned, dtype=np.float64)
                left_encoder_ts = np.zeros(n_aligned, dtype=np.float64)

                right_angles = np.zeros(n_aligned, dtype=np.float32)
                right_timestamps = np.zeros(n_aligned, dtype=np.float64)
                right_stereo_ts = np.zeros(n_aligned, dtype=np.float64)
                right_mono_ts = np.zeros(n_aligned, dtype=np.float64)
                right_encoder_ts = np.zeros(n_aligned, dtype=np.float64)

                sync_timestamps = np.zeros(n_aligned, dtype=np.float64)

                for idx, (l_idx, r_idx_pair) in enumerate(pairs):
                    left_stereo_ds[idx] = left_f['stereo_jpeg'][l_idx]
                    left_mono_ds[idx] = left_f['mono_jpeg'][l_idx]
                    right_stereo_ds[idx] = right_f['stereo_jpeg'][r_idx_pair]
                    right_mono_ds[idx] = right_f['mono_jpeg'][r_idx_pair]

                    left_angles[idx] = float(left_f['angles'][l_idx])
                    left_timestamps[idx] = float(left_f['timestamps'][l_idx])
                    left_stereo_ts[idx] = float(left_f['stereo_timestamps'][l_idx])
                    left_mono_ts[idx] = float(left_f['mono_timestamps'][l_idx])
                    left_encoder_ts[idx] = float(left_f['encoder_timestamps'][l_idx])

                    right_angles[idx] = float(right_f['angles'][r_idx_pair])
                    right_timestamps[idx] = float(right_f['timestamps'][r_idx_pair])
                    right_stereo_ts[idx] = float(right_f['stereo_timestamps'][r_idx_pair])
                    right_mono_ts[idx] = float(right_f['mono_timestamps'][r_idx_pair])
                    right_encoder_ts[idx] = float(right_f['encoder_timestamps'][r_idx_pair])

                    sync_timestamps[idx] = (left_timestamps[idx] + right_timestamps[idx]) / 2.0

                out_f.create_dataset('left_angles', data=left_angles, dtype=np.float32)
                out_f.create_dataset('left_timestamps', data=left_timestamps, dtype=np.float64)
                out_f.create_dataset('left_stereo_timestamps', data=left_stereo_ts, dtype=np.float64)
                out_f.create_dataset('left_mono_timestamps', data=left_mono_ts, dtype=np.float64)
                out_f.create_dataset('left_encoder_timestamps', data=left_encoder_ts, dtype=np.float64)

                out_f.create_dataset('right_angles', data=right_angles, dtype=np.float32)
                out_f.create_dataset('right_timestamps', data=right_timestamps, dtype=np.float64)
                out_f.create_dataset('right_stereo_timestamps', data=right_stereo_ts, dtype=np.float64)
                out_f.create_dataset('right_mono_timestamps', data=right_mono_ts, dtype=np.float64)
                out_f.create_dataset('right_encoder_timestamps', data=right_encoder_ts, dtype=np.float64)

                out_f.create_dataset('sync_timestamps', data=sync_timestamps, dtype=np.float64)

        os.remove(left_path)
        os.remove(right_path)

        print(f"✅ 双手数据合并完成: {merged_path}")
        print(f"  对齐帧数: {n_aligned}")
        return merged_path
    except Exception as exc:
        print(f"❌ 合并双手数据失败: {exc}")
        return None


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
    parser.add_argument("--realtime-write", dest="realtime_write", action="store_true",
                       help="启用实时写入模式（录制时直接写入磁盘，默认开启）")
    parser.add_argument("--no-realtime-write", dest="realtime_write", action="store_false",
                       help="禁用实时写入模式（改为内存缓存后离线保存）")
    parser.set_defaults(realtime_write=True)
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
            collector = DualHandCollector(
                config_path,
                enable_realtime_write=args.realtime_write
            )
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
                result = collector.stop_recording()
                save_cfg = collector.config.get('save', {})
                output_dir = save_cfg.get('output_dir', './data')
                jpeg_quality = save_cfg.get('jpeg_quality', 85)
                if args.realtime_write:
                    left_path, right_path = result if isinstance(result, tuple) else (None, None)
                    merged_path = merge_dual_hand_files(left_path, right_path, collector.output_dir)
                    if merged_path:
                        print(f"\n✅ 双手数据已合并保存到: {merged_path}")
                else:
                    if result:
                        save_dual_hand_data(result, output_dir, jpeg_quality)
            collector.stop()
        
        else:
            # 单手模式
            config = yaml.safe_load(open(config_path, 'r'))
            hand_name = args.mode.upper()
            hand_config = config.get(f'{args.mode}_hand', {})
            
            if not hand_config:
                print(f"❌ 配置文件中没有找到 {args.mode}_hand 配置")
                sys.exit(1)
            
            save_cfg = config.get('save', {})
            output_dir = save_cfg.get('output_dir', './data')
            jpeg_quality = save_cfg.get('jpeg_quality', 85)
            collector = HandCollector(
                hand_config,
                hand_name,
                enable_realtime_write=args.realtime_write,
                output_dir=output_dir,
                jpeg_quality=jpeg_quality
            )
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
                if args.realtime_write:
                    if isinstance(data, str) and data:
                        print(f"\n[{hand_name}] ✅ 数据已实时保存到: {data}")
                else:
                    if isinstance(data, list) and data:
                        print(f"\n录制完成: {len(data)} 帧")
                        print("数据已收集，可扩展保存功能")
            
            collector.stop()
        
    except KeyboardInterrupt:
        print("\n中断")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n错误: {e}")


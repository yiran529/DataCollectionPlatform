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


from hand_collector import HandFrame, HandCollector, visualize_hand


@dataclass
class DualHandFrame:
    """双手同步帧"""
    left: HandFrame
    right: HandFrame
    timestamp: float
    idx: int



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



def visualize_dual_hand(collector: DualHandCollector):
    """可视化双手数据"""
    print("\n双手可视化（按 'q' 退出）...")
    
    window_name = "Dual Hand - Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    left_angle_history = []
    right_angle_history = []
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
            left_frame = collector.left.get_current_frame() if collector.left else None
            right_frame = collector.right.get_current_frame() if collector.right else None
            
            if left_frame is None and right_frame is None:
                time.sleep(0.01)
                continue
            
            # 获取内存信息
            mem_text = ""
            if process:
                try:
                    mem_mb = process.memory_info().rss / (1024 * 1024)
                    mem_delta = mem_mb - initial_memory_mb
                    mem_text = f"Memory: {mem_mb:.0f}MB (+{mem_delta:.0f}MB)"
                except:
                    pass
            
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
                cv2.putText(angle_img, f"L: {angle:.2f}", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示内存（仅在左侧显示一次）
                if mem_text:
                    cv2.putText(angle_img, mem_text, (10, 55), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
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
                cv2.putText(angle_img, f"R: {angle:.2f}", (10, 25), 
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


#!/usr/bin/env python3
"""
单手数据收集器

收集单手的同步数据：
- 单目相机
- 双目相机
- 角度编码器

默认收集右手数据，可通过 --hand 参数选择左手或右手
"""

import sys
import os
import cv2
import numpy as np
import time
import yaml
from typing import List

# 导入dual_hand_collector中的类
from dual_hand_collector import HandCollector, HandFrame, visualize_hand

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def save_single_hand_data(data: List[HandFrame], hand_name: str, output_dir: str,
                          jpeg_quality: int = 85) -> str:
    """保存单手数据到HDF5（流式写入优化内存）"""
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
    hand_lower = hand_name.lower()
    filename = f"{prefix}_{hand_lower}_hand_data.h5"
    filepath = os.path.join(output_dir, filename)
    
    n_frames = len(data)
    print(f"\n保存{hand_name}手数据: {filepath}")
    print(f"  帧数: {n_frames}")
    
    start_time = time.time()
    
    # 流式写入HDF5（避免一次性加载所有数据到内存）
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    
    print("  写入HDF5（流式）...")
    write_start = time.time()
    
    with h5py.File(filepath, 'w', libver='latest') as f:
        # 元数据
        f.attrs['n_frames'] = n_frames
        f.attrs['hand'] = hand_lower
        f.attrs['stereo_shape'] = data[0].stereo.shape
        f.attrs['mono_shape'] = data[0].mono.shape
        f.attrs['jpeg_quality'] = jpeg_quality
        f.attrs['created_at'] = datetime.now().isoformat()
        
        # 创建数据集（使用可变长度数据类型）
        dt = h5py.special_dtype(vlen=np.uint8)
        
        # 创建数据集（预先分配空间）
        stereo_ds = f.create_dataset('stereo_jpeg', (n_frames,), dtype=dt)
        mono_ds = f.create_dataset('mono_jpeg', (n_frames,), dtype=dt)
        
        angles = np.zeros(n_frames, dtype=np.float32)
        timestamps = np.zeros(n_frames, dtype=np.float64)
        stereo_timestamps = np.zeros(n_frames, dtype=np.float64)
        mono_timestamps = np.zeros(n_frames, dtype=np.float64)
        encoder_timestamps = np.zeros(n_frames, dtype=np.float64)
        
        # 分批处理和写入（每批100帧）
        batch_size = 100
        for i in range(0, n_frames, batch_size):
            batch_end = min(i + batch_size, n_frames)
            batch_data = data[i:batch_end]
            
            for j, frame in enumerate(batch_data):
                idx = i + j
                
                # 压缩图像（立即写入，不保存到列表）
                success_s, s_jpeg = cv2.imencode('.jpg', frame.stereo, encode_params)
                success_m, m_jpeg = cv2.imencode('.jpg', frame.mono, encode_params)
                
                if not (success_s and success_m):
                    print(f"\n⚠️ 警告: 第 {idx} 帧图像压缩失败，跳过")
                    continue
                
                # 对于可变长度数据类型，直接传递numpy数组（h5py会自动处理）
                stereo_ds[idx] = s_jpeg
                mono_ds[idx] = m_jpeg
                
                angles[idx] = frame.angle
                timestamps[idx] = frame.timestamp
                stereo_timestamps[idx] = frame.stereo_ts
                mono_timestamps[idx] = frame.mono_ts
                encoder_timestamps[idx] = frame.encoder_ts
            
            # 显示进度
            progress = (batch_end / n_frames) * 100
            print(f"  进度: {batch_end}/{n_frames} ({progress:.1f}%)", end='\r')
        
        print()  # 换行
        
        # 写入角度和时间戳数据
        f.create_dataset('angles', data=angles, dtype=np.float32)
        f.create_dataset('timestamps', data=timestamps, dtype=np.float64)
        f.create_dataset('stereo_timestamps', data=stereo_timestamps, dtype=np.float64)
        f.create_dataset('mono_timestamps', data=mono_timestamps, dtype=np.float64)
        f.create_dataset('encoder_timestamps', data=encoder_timestamps, dtype=np.float64)
    
    write_time = time.time() - write_start
    total_time = time.time() - start_time
    
    file_size = os.path.getsize(filepath) / (1024 * 1024)
    
    print(f"  写入耗时: {write_time:.2f}s")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  文件大小: {file_size:.1f}MB")
    print(f"✅ 保存完成: {filepath}")
    
    return filepath


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="单手数据收集器（默认右手）")
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                       help="配置文件路径")
    parser.add_argument("--hand", "-H", type=str, choices=['left', 'right'],
                       default='right', help="选择左手或右手（默认：right）")
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
        # 加载配置
        config = yaml.safe_load(open(config_path, 'r'))
        hand_name = args.hand.upper()
        hand_config = config.get(f'{args.hand}_hand', {})
        
        if not hand_config:
            print(f"❌ 配置文件中没有找到 {args.hand}_hand 配置")
            sys.exit(1)
        
        # 创建收集器
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
            # 可视化模式
            visualize_hand(collector, hand_name)
        
        elif args.record:
            # 录制模式
            print(f"\n[{hand_name}] 按回车键开始录制...")
            input()
            collector.start_recording()
            print(f"[{hand_name}] 录制中... 按回车键停止录制")
            input()
            data = collector.stop_recording()
            
            if data:
                print(f"\n[{hand_name}] 录制完成: {len(data)} 帧")
                save_cfg = config.get('save', {})
                output_dir = save_cfg.get('output_dir', './data')
                jpeg_quality = save_cfg.get('jpeg_quality', 85)
                save_single_hand_data(data, hand_name, output_dir, jpeg_quality)
        
        collector.stop()
        
    except KeyboardInterrupt:
        print("\n中断")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n错误: {e}")


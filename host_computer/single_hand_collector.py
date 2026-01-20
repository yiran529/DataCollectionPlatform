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

# 导入hand_collector中的类
from hand_collector import HandCollector, HandFrame, visualize_hand

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def save_single_hand_data(temp_h5_path: str, aligned_indices: List, hand_name: str, 
                          output_dir: str) -> str:
    """保存单手数据到HDF5（直接复制临时文件中的JPEG数据）"""
    if not aligned_indices:
        print("❌ 无对齐索引")
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
    
    n_frames = len(aligned_indices)
    print(f"\n保存{hand_name}手数据: {filepath}")
    print(f"  帧数: {n_frames}")
    
    start_time = time.time()
    
    try:
        with h5py.File(temp_h5_path, 'r') as src:
            with h5py.File(filepath, 'w') as dst:
                # 读取源数据集
                src_stereo_jpeg = src['stereo_jpeg']
                src_mono_jpeg = src['mono_jpeg']
                src_stereo_ts = src['stereo_timestamps']
                src_mono_ts = src['mono_timestamps']
                src_encoder_angles = src['encoder_angles']
                src_encoder_ts = src['encoder_timestamps']
                
                # 创建目标数据集
                dst_stereo = dst.create_dataset('stereo_jpeg', 
                                                shape=(n_frames,), 
                                                dtype=h5py.vlen_dtype(np.uint8))
                dst_mono = dst.create_dataset('mono_jpeg', 
                                              shape=(n_frames,), 
                                              dtype=h5py.vlen_dtype(np.uint8))
                dst_angles = dst.create_dataset('angles', 
                                                shape=(n_frames,), 
                                                dtype=np.float32)
                dst_timestamps = dst.create_dataset('timestamps', 
                                                    shape=(n_frames,), 
                                                    dtype=np.float64)
                dst_stereo_ts = dst.create_dataset('stereo_timestamps', 
                                                   shape=(n_frames,), 
                                                   dtype=np.float64)
                dst_mono_ts = dst.create_dataset('mono_timestamps', 
                                                 shape=(n_frames,), 
                                                 dtype=np.float64)
                dst_encoder_ts = dst.create_dataset('encoder_timestamps', 
                                                    shape=(n_frames,), 
                                                    dtype=np.float64)
                
                # 复制对齐的数据
                for i, (s_idx, m_idx, e_idx) in enumerate(aligned_indices):
                    dst_stereo[i] = src_stereo_jpeg[s_idx]
                    dst_mono[i] = src_mono_jpeg[m_idx]
                    dst_angles[i] = src_encoder_angles[e_idx]
                    dst_timestamps[i] = src_stereo_ts[s_idx]  # 使用stereo时间戳作为主时间戳
                    dst_stereo_ts[i] = src_stereo_ts[s_idx]
                    dst_mono_ts[i] = src_mono_ts[m_idx]
                    dst_encoder_ts[i] = src_encoder_ts[e_idx]
                    
                    if (i + 1) % 100 == 0 or i == n_frames - 1:
                        print(f"  保存进度: {i+1}/{n_frames}", end='\r')
                
                print()  # 换行
                
                # 添加元数据
                dst.attrs['n_frames'] = n_frames
                dst.attrs['hand'] = hand_lower
                dst.attrs['created_at'] = datetime.now().isoformat()
                
                # 从源复制分辨率等元数据
                if 'stereo_resolution' in src.attrs:
                    dst.attrs['stereo_resolution'] = src.attrs['stereo_resolution']
                if 'mono_resolution' in src.attrs:
                    dst.attrs['mono_resolution'] = src.attrs['mono_resolution']
                if 'jpeg_quality' in src.attrs:
                    dst.attrs['jpeg_quality'] = src.attrs['jpeg_quality']
        
        total_time = time.time() - start_time
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        
        print(f"  总耗时: {total_time:.2f}s")
        print(f"  文件大小: {file_size:.1f}MB")
        print(f"✅ 保存完成: {filepath}")
        
        return filepath
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        import traceback
        traceback.print_exc()
        return None


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
            
            # 停止录制并获取临时HDF5路径
            temp_h5_path = collector.stop_recording()
            
            if temp_h5_path:
                # 计算对齐索引
                aligned_indices = collector.align_and_get_indices(max_time_diff_ms=200.0)
                
                if aligned_indices:
                    print(f"\n[{hand_name}] 对齐完成: {len(aligned_indices)} 帧")
                    save_cfg = config.get('save', {})
                    output_dir = save_cfg.get('output_dir', './data')
                    
                    # 保存数据
                    saved_path = save_single_hand_data(temp_h5_path, aligned_indices, 
                                                      hand_name, output_dir)
                    
                    # 清理临时文件
                    collector.cleanup_temp_file()
                else:
                    print(f"\n[{hand_name}] 警告: 对齐失败")
            else:
                print(f"\n[{hand_name}] 警告: 录制失败")
        
        collector.stop()
        
    except KeyboardInterrupt:
        print("\n中断")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n错误: {e}")


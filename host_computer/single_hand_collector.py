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


def save_single_hand_data(data: List[HandFrame], hand_name: str, output_dir: str,
                          jpeg_quality: int = 85) -> str:
    """保存单手数据到HDF5（真正的流式写入，优化内存）"""
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
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    
    print("  流式写入HDF5（逐批处理以节省内存）...")
    write_start = time.time()
    
    with h5py.File(filepath, 'w', libver='latest') as f:
        # 元数据
        f.attrs['n_frames'] = n_frames
        f.attrs['hand'] = hand_lower
        f.attrs['stereo_shape'] = data[0].stereo.shape
        f.attrs['mono_shape'] = data[0].mono.shape
        f.attrs['jpeg_quality'] = jpeg_quality
        f.attrs['created_at'] = datetime.now().isoformat()
        
        # 创建可变长度数据集（初始大小为0）
        dt = h5py.special_dtype(vlen=np.uint8)
        stereo_dset = f.create_dataset('stereo_jpeg', (0,), dtype=dt, maxshape=(None,))
        mono_dset = f.create_dataset('mono_jpeg', (0,), dtype=dt, maxshape=(None,))
        
        # 创建固定长度数据集
        angles_dset = f.create_dataset('angles', (0,), dtype=np.float32, maxshape=(None,))
        timestamps_dset = f.create_dataset('timestamps', (0,), dtype=np.float64, maxshape=(None,))
        stereo_ts_dset = f.create_dataset('stereo_timestamps', (0,), dtype=np.float64, maxshape=(None,))
        mono_ts_dset = f.create_dataset('mono_timestamps', (0,), dtype=np.float64, maxshape=(None,))
        encoder_ts_dset = f.create_dataset('encoder_timestamps', (0,), dtype=np.float64, maxshape=(None,))
        
        # 逐批处理并写入（每批10帧以最小化内存占用）
        batch_size = 10
        written_count = 0
        
        for i in range(0, n_frames, batch_size):
            batch_end = min(i + batch_size, n_frames)
            batch_data = data[i:batch_end]
            
            # 临时存储当前批次数据
            batch_stereo_jpegs = []
            batch_mono_jpegs = []
            batch_angles = []
            batch_timestamps = []
            batch_stereo_ts = []
            batch_mono_ts = []
            batch_encoder_ts = []
            
            for frame in batch_data:
                # 压缩图像
                success_s, s_jpeg = cv2.imencode('.jpg', frame.stereo, encode_params)
                success_m, m_jpeg = cv2.imencode('.jpg', frame.mono, encode_params)
                
                if not (success_s and success_m):
                    print(f"\n⚠️ 警告: 图像压缩失败，跳过帧")
                    continue
                
                # 添加到当前批次
                batch_stereo_jpegs.append(np.asarray(s_jpeg, dtype=np.uint8))
                batch_mono_jpegs.append(np.asarray(m_jpeg, dtype=np.uint8))
                batch_angles.append(frame.angle)
                batch_timestamps.append(frame.timestamp)
                batch_stereo_ts.append(frame.stereo_ts)
                batch_mono_ts.append(frame.mono_ts)
                batch_encoder_ts.append(frame.encoder_ts)
            
            # 写入当前批次到HDF5
            if batch_stereo_jpegs:
                batch_len = len(batch_stereo_jpegs)
                new_size = written_count + batch_len
                
                # 扩展数据集
                stereo_dset.resize((new_size,))
                mono_dset.resize((new_size,))
                angles_dset.resize((new_size,))
                timestamps_dset.resize((new_size,))
                stereo_ts_dset.resize((new_size,))
                mono_ts_dset.resize((new_size,))
                encoder_ts_dset.resize((new_size,))
                
                # 写入数据
                stereo_dset[written_count:new_size] = batch_stereo_jpegs
                mono_dset[written_count:new_size] = batch_mono_jpegs
                angles_dset[written_count:new_size] = batch_angles
                timestamps_dset[written_count:new_size] = batch_timestamps
                stereo_ts_dset[written_count:new_size] = batch_stereo_ts
                mono_ts_dset[written_count:new_size] = batch_mono_ts
                encoder_ts_dset[written_count:new_size] = batch_encoder_ts
                
                written_count = new_size
                
                # 清空批次数据释放内存
                del batch_stereo_jpegs, batch_mono_jpegs
                del batch_angles, batch_timestamps
                del batch_stereo_ts, batch_mono_ts, batch_encoder_ts
            
            # 显示进度
            progress = (batch_end / n_frames) * 100
            print(f"  写入进度: {written_count}/{n_frames} ({progress:.1f}%)", end='\r')
    
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
    # parser.add_argument("--realtime-write", action="store_true", default=True, 
    #                    help="启用实时写入模式（录制时直接写入磁盘，避免OOM）")
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
        
        # 获取保存配置
        save_cfg = config.get('save', {})
        output_dir = save_cfg.get('output_dir', './data')
        jpeg_quality = save_cfg.get('jpeg_quality', 85)
        
        # 创建收集器（支持实时写入）
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
            # 可视化模式
            visualize_hand(collector, hand_name)
        
        elif args.record:
            # 录制模式
            mode_text = "实时写入" if args.realtime_write else "内存缓存"
            print(f"\n[{hand_name}] 按回车键开始录制（{mode_text}模式）...")
            input()
            collector.start_recording()
            print(f"[{hand_name}] 录制中... 按回车键停止录制")
            input()
            result = collector.stop_recording()
            
            if args.realtime_write:
                # 实时写入模式：返回文件路径
                if result:
                    print(f"\n[{hand_name}] ✅ 数据已保存到: {result}")
            else:
                # 内存模式：返回数据列表，需要保存
                if result:
                    print(f"\n[{hand_name}] 录制完成: {len(result)} 帧")
                    save_single_hand_data(result, hand_name, output_dir, jpeg_quality)
        
        collector.stop()
        
    except KeyboardInterrupt:
        print("\n中断")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n错误: {e}")


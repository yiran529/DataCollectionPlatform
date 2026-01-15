#!/usr/bin/env python3
"""
分析时间戳文件，计算稳定后的偏移量
"""

import numpy as np
import argparse
import os


def analyze_timestamps(filepath: str, skip_warmup: int = 100):
    """
    分析时间戳文件
    
    Args:
        filepath: timestamps.txt 路径
        skip_warmup: 跳过前N帧（预热阶段）
    """
    # 读取数据
    frame_idx = []
    stereo_ts = []
    mono_ts = []
    time_diff = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split(',')
            if len(parts) >= 4:
                frame_idx.append(int(parts[0]))
                stereo_ts.append(float(parts[1]))
                mono_ts.append(float(parts[2]))
                time_diff.append(float(parts[3]))
    
    time_diff = np.array(time_diff)
    total_frames = len(time_diff)
    
    print("=" * 60)
    print("时间戳分析结果")
    print("=" * 60)
    print(f"总帧数: {total_frames}")
    print(f"预热跳过: {skip_warmup} 帧")
    
    # 全部数据统计
    print(f"\n【全部数据】")
    print(f"  平均时间差: {np.mean(time_diff):.2f} ms")
    print(f"  标准差: {np.std(time_diff):.2f} ms")
    print(f"  范围: {np.min(time_diff):.2f} ~ {np.max(time_diff):.2f} ms")
    
    # 稳定后数据统计
    if total_frames > skip_warmup:
        stable_diff = time_diff[skip_warmup:]
        print(f"\n【稳定后数据】(跳过前 {skip_warmup} 帧)")
        print(f"  帧数: {len(stable_diff)}")
        print(f"  平均时间差: {np.mean(stable_diff):.2f} ms")
        print(f"  标准差: {np.std(stable_diff):.2f} ms")
        print(f"  范围: {np.min(stable_diff):.2f} ~ {np.max(stable_diff):.2f} ms")
        
        # 建议的固定偏移
        recommended_offset = np.mean(stable_diff)
        print(f"\n【建议配置】")
        print(f"  固定偏移量: {recommended_offset:.2f} ms")
        print(f"  (立体相机比单目相机慢 {recommended_offset:.0f} ms)")
        
        # 应用偏移后的误差
        corrected_diff = stable_diff - recommended_offset
        print(f"\n【应用偏移后】")
        print(f"  平均误差: {np.mean(corrected_diff):.2f} ms")
        print(f"  标准差: {np.std(corrected_diff):.2f} ms")
        print(f"  范围: {np.min(corrected_diff):.2f} ~ {np.max(corrected_diff):.2f} ms")
        
        # 输出可用于配置的值
        print(f"\n" + "=" * 60)
        print(f"复制以下值用于配置:")
        print(f"  STEREO_MONO_OFFSET_MS = {recommended_offset:.1f}")
        print("=" * 60)
        
        return recommended_offset
    else:
        print(f"\n数据不足，无法分析稳定状态")
        return None


def main():
    parser = argparse.ArgumentParser(description="分析双摄像头时间戳")
    parser.add_argument("filepath", help="timestamps.txt 文件路径")
    parser.add_argument("--skip", type=int, default=100, help="跳过预热帧数")
    args = parser.parse_args()
    
    if not os.path.exists(args.filepath):
        print(f"文件不存在: {args.filepath}")
        return
    
    analyze_timestamps(args.filepath, args.skip)


if __name__ == "__main__":
    main()


















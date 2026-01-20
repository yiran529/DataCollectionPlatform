#!/usr/bin/env python3
"""
测试流式HDF5实现的内存使用

验证：
1. 录制过程中内存保持稳定（不随时间增长）
2. 临时HDF5文件正确创建和写入
3. 对齐索引计算正确
4. 最终保存不需要重新编码JPEG
"""

import os
import sys
import time
import psutil
import traceback

# 测试配置
TEST_DURATION = 10  # 录制10秒测试
MEMORY_SAMPLES = []


def monitor_memory(process, interval=0.5, duration=10):
    """监控内存使用"""
    start = time.time()
    samples = []
    
    while time.time() - start < duration:
        try:
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / 1024 / 1024
            samples.append((time.time() - start, rss_mb))
            time.sleep(interval)
        except:
            break
    
    return samples


def analyze_memory_growth(samples):
    """分析内存增长趋势"""
    if len(samples) < 5:
        return None, None
    
    # 计算前后两段的平均值
    n = len(samples)
    first_half = [s[1] for s in samples[:n//2]]
    second_half = [s[1] for s in samples[n//2:]]
    
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    
    growth_mb = avg_second - avg_first
    growth_rate = growth_mb / (samples[-1][0] - samples[0][0])  # MB/s
    
    return growth_mb, growth_rate


def test_streaming_recording():
    """测试流式录制"""
    print("=" * 60)
    print("流式HDF5录制测试")
    print("=" * 60)
    
    # 导入（延迟导入以确保路径正确）
    from hand_collector import HandCollector
    import yaml
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    config = yaml.safe_load(open(config_path, 'r'))
    hand_config = config.get('right_hand', {})
    
    if not hand_config:
        print("❌ 配置文件中没有right_hand配置")
        return False
    
    print("\n1. 初始化HandCollector...")
    collector = HandCollector(hand_config, 'RIGHT')
    collector.start()
    
    if not collector.wait_ready(timeout=10):
        print("❌ 初始化失败")
        return False
    
    print("✅ 初始化成功")
    
    # 预热
    print("\n2. 预热和校准...")
    collector.warmup_and_calibrate(1.0, 0.5)
    print("✅ 预热完成")
    
    # 开始录制
    print(f"\n3. 开始录制 ({TEST_DURATION}秒)...")
    collector.start_recording()
    
    # 监控内存
    process = psutil.Process()
    initial_mem = process.memory_info().rss / 1024 / 1024
    print(f"初始内存: {initial_mem:.1f} MB")
    
    # 记录过程中内存
    start_time = time.time()
    mem_samples = []
    
    while time.time() - start_time < TEST_DURATION:
        elapsed = time.time() - start_time
        mem = process.memory_info().rss / 1024 / 1024
        mem_samples.append((elapsed, mem))
        
        # 每秒报告一次
        if int(elapsed) != int(elapsed - 0.5):
            print(f"  录制中... {elapsed:.1f}s, 内存: {mem:.1f} MB", end='\r')
        
        time.sleep(0.5)
    
    print()
    
    # 停止录制
    print("\n4. 停止录制...")
    temp_h5_path = collector.stop_recording()
    
    if not temp_h5_path or not os.path.exists(temp_h5_path):
        print("❌ 录制失败，没有生成临时HDF5文件")
        collector.stop()
        return False
    
    # 检查文件大小
    file_size_mb = os.path.getsize(temp_h5_path) / 1024 / 1024
    print(f"✅ 临时HDF5文件: {temp_h5_path}")
    print(f"   文件大小: {file_size_mb:.1f} MB")
    
    # 分析内存使用
    print("\n5. 内存使用分析:")
    final_mem = process.memory_info().rss / 1024 / 1024
    peak_mem = max(s[1] for s in mem_samples)
    
    print(f"  初始内存: {initial_mem:.1f} MB")
    print(f"  峰值内存: {peak_mem:.1f} MB")
    print(f"  最终内存: {final_mem:.1f} MB")
    print(f"  内存增长: {final_mem - initial_mem:.1f} MB")
    
    # 分析增长趋势
    growth_mb, growth_rate = analyze_memory_growth(mem_samples)
    if growth_mb is not None:
        print(f"  增长趋势: {growth_mb:.1f} MB ({growth_rate:.2f} MB/s)")
        
        # 判断是否合理（流式应该增长很少）
        if growth_mb < 50:  # 小于50MB增长认为正常
            print("  ✅ 内存使用稳定（流式写入生效）")
        else:
            print(f"  ⚠️ 内存增长较大，可能存在问题")
    
    # 计算对齐索引
    print("\n6. 计算对齐索引...")
    aligned_indices = collector.align_and_get_indices(max_time_diff_ms=200.0)
    
    if not aligned_indices:
        print("❌ 对齐失败")
        collector.cleanup_temp_file()
        collector.stop()
        return False
    
    print(f"✅ 对齐完成: {len(aligned_indices)} 帧")
    
    # 保存最终数据
    print("\n7. 保存最终HDF5文件...")
    output_path = os.path.join(os.path.dirname(__file__), 'test_streaming_output.h5')
    
    if collector.save_to_hdf5(output_path, aligned_indices):
        output_size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"✅ 保存成功: {output_path}")
        print(f"   文件大小: {output_size_mb:.1f} MB")
        
        # 验证文件大小接近（说明没有重新编码）
        size_diff = abs(output_size_mb - file_size_mb)
        if size_diff < file_size_mb * 0.2:  # 差异小于20%
            print("  ✅ 文件大小接近临时文件（无重复编码）")
        else:
            print(f"  ⚠️ 文件大小差异较大: {size_diff:.1f} MB")
    else:
        print("❌ 保存失败")
    
    # 清理
    collector.cleanup_temp_file()
    collector.stop()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_streaming_recording()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 测试失败: {e}")
        traceback.print_exc()
        sys.exit(1)

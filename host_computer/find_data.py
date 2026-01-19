#!/usr/bin/env python3
"""
查找数据文件和诊断数据保存问题
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("数据文件诊断工具")
print("=" * 70)

# 获取项目目录
project_dir = Path.home() / "DataCollectionPlatform"
host_dir = project_dir / "host_computer"

print(f"\n项目目录: {project_dir}")
print(f"主程序目录: {host_dir}")

# 检查各个可能的数据目录
data_dirs = [
    project_dir / "data",
    host_dir / "data",
    Path.cwd() / "data",
    project_dir / "host_computer" / "data",
]

print("\n" + "=" * 70)
print("检查可能的数据目录")
print("=" * 70)

found_files = []

for data_dir in data_dirs:
    print(f"\n检查: {data_dir}")
    
    if data_dir.exists():
        print(f"  ✓ 目录存在")
        files = list(data_dir.glob("*_hand_data.h5"))
        if files:
            print(f"  ✓ 找到 {len(files)} 个数据文件：")
            for f in sorted(files, reverse=True):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"    - {f.name} ({size_mb:.1f} MB)")
                found_files.append(f)
        else:
            print(f"  ⚠️ 目录为空（没有 *_hand_data.h5 文件）")
    else:
        print(f"  ✗ 目录不存在")

# 检查 config.yaml 中的配置
print("\n" + "=" * 70)
print("检查配置文件")
print("=" * 70)

config_file = host_dir / "config.yaml"
print(f"\n配置文件: {config_file}")

if config_file.exists():
    print("  ✓ 文件存在")
    with open(config_file, 'r') as f:
        content = f.read()
        if 'save:' in content:
            print("  ✓ 包含 save 配置")
            # 提取 save 配置
            in_save = False
            for line in content.split('\n'):
                if 'save:' in line:
                    in_save = True
                if in_save:
                    print(f"    {line}")
                    if line.strip() and not line.startswith(' ' * 4) and 'save:' not in line:
                        break
        else:
            print("  ⚠️ 未包含 save 配置（将使用默认的 ./data）")
else:
    print("  ✗ 配置文件不存在")

# 总结
print("\n" + "=" * 70)
print("诊断结果")
print("=" * 70)

if found_files:
    print(f"\n✓ 找到 {len(found_files)} 个数据文件")
    print(f"最新文件: {found_files[0]}")
else:
    print("\n❌ 未找到数据文件")
    print("\n可能原因：")
    print("  1. 时间戳同步失败（对齐后: 0 帧）")
    print("     → 这是正常的，说明 Stereo 和 Mono 帧率差异大")
    print("  2. 数据保存目录未创建")
    print("     → 尝试手动创建: mkdir -p ~/DataCollectionPlatform/data")
    print("  3. 没有进行任何录制")
    print("     → 确认已按回车开始和停止录制")
    print("\n解决方案：")
    print("  1. 修改 config.yaml，添加 save 配置：")
    print("     save:")
    print("       output_dir: ./data")
    print("  2. 重新运行采集程序")
    print("  3. 检查是否有权限写入目录")

print("\n" + "=" * 70)

#!/usr/bin/env python3
"""
Jetson Xavier 专用诊断脚本
测试不同来源的 NumPy 和 OpenCV
"""

import sys
import os
import subprocess

print("=" * 70)
print("Jetson Xavier NumPy/OpenCV 诊断工具")
print("=" * 70)

# 检测平台
print("\n[1/5] 检测平台...")
if os.path.exists('/etc/nv_tegra_release'):
    with open('/etc/nv_tegra_release', 'r') as f:
        jetson_info = f.read().strip()
    print(f"  ✓ 检测到 Jetson 平台")
    print(f"  信息: {jetson_info}")
else:
    print("  ✗ 未检测到 Jetson 平台")
    sys.exit(1)

# 检查 Python 版本
print("\n[2/5] Python 信息...")
print(f"  Python 版本: {sys.version}")
print(f"  Python 路径: {sys.executable}")

# 检查 sys.path
print("\n[3/5] 检查 Python 路径...")
for i, path in enumerate(sys.path[:5], 1):
    print(f"  {i}. {path}")

# 尝试导入系统 NumPy（使用 subprocess 避免崩溃传播）
print("\n[4/5] 测试系统 NumPy...")
result = subprocess.run(
    [sys.executable, "-c", 
     "import sys; sys.path.insert(0, '/usr/lib/python3/dist-packages'); "
     "import numpy; print('NumPy 版本:', numpy.__version__)"],
    capture_output=True,
    text=True,
    timeout=5
)

if result.returncode == 0:
    print(f"  ✓ 系统 NumPy 可用")
    print(f"  {result.stdout.strip()}")
else:
    print(f"  ✗ 系统 NumPy 导入失败")
    if "Illegal instruction" in result.stderr:
        print(f"  错误: Illegal instruction - CPU 不支持编译的指令集")
    else:
        print(f"  错误: {result.stderr[:100]}")

# 尝试导入系统 OpenCV
print("\n[5/5] 测试系统 OpenCV...")
result = subprocess.run(
    [sys.executable, "-c", 
     "import sys; sys.path.insert(0, '/usr/lib/python3/dist-packages'); "
     "import cv2; print('OpenCV 版本:', cv2.__version__)"],
    capture_output=True,
    text=True,
    timeout=5
)

if result.returncode == 0:
    print(f"  ✓ 系统 OpenCV 可用")
    print(f"  {result.stdout.strip()}")
else:
    print(f"  ✗ 系统 OpenCV 导入失败")
    if "Illegal instruction" in result.stderr:
        print(f"  错误: Illegal instruction - CPU 不支持编译的指令集")
    else:
        print(f"  错误: {result.stderr[:100]}")

print("\n" + "=" * 70)
print("诊断结果分析")
print("=" * 70)

print("\n📋 解决方案：")
print("")
print("1️⃣  如果系统包导入成功，运行以下命令切换到系统包：")
print("")
print("    pip uninstall -y numpy opencv-python opencv-contrib-python")
print("    export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH")
print("    python3 diagnose_crash.py  # 验证")
print("")
print("2️⃣  如果系统包也崩溃，问题在于 Jetson 系统本身：")
print("")
print("    a) 重新安装系统包:")
print("       sudo apt install --reinstall python3-opencv python3-numpy")
print("")
print("    b) 使用环境变量禁用 AVX2:")
print("       export OPENBLAS_CORETYPE=ARMV8")
print("       python3 diagnose_crash.py")
print("")
print("3️⃣  如果以上都失败，升级 JetPack：")
print("")
print("    sudo apt update && sudo apt upgrade -y")
print("    # 或重新刷 JetPack 系统")
print("")

print("=" * 70)

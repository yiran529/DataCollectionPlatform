#!/usr/bin/env python3
"""
诊断 Illegal Instruction 错误
逐步加载各个模块，找出导致崩溃的具体库
"""

import sys
import os

print("=" * 60)
print("开始诊断 Illegal Instruction 错误")
print("=" * 60)

# 测试1: Python基础
print("\n[1/10] 测试 Python 基础...")
try:
    print(f"  Python版本: {sys.version}")
    print("  ✓ Python 基础正常")
except Exception as e:
    print(f"  ✗ Python 基础错误: {e}")
    sys.exit(1)

# 测试2: NumPy
print("\n[2/10] 测试 NumPy...")
try:
    import numpy as np
    print(f"  NumPy 版本: {np.__version__}")
    # 测试基本运算
    a = np.array([1, 2, 3])
    b = np.dot(a, a)
    print(f"  测试运算: {b}")
    print("  ✓ NumPy 正常")
except Exception as e:
    print(f"  ✗ NumPy 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: OpenCV
print("\n[3/10] 测试 OpenCV...")
try:
    import cv2
    print(f"  OpenCV 版本: {cv2.__version__}")
    print("  ✓ OpenCV 导入正常")
except Exception as e:
    print(f"  ✗ OpenCV 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: OpenCV 基本操作
print("\n[4/10] 测试 OpenCV 基本操作...")
try:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  测试图像转换: shape={gray.shape}")
    print("  ✓ OpenCV 基本操作正常")
except Exception as e:
    print(f"  ✗ OpenCV 基本操作错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: YAML
print("\n[5/10] 测试 YAML...")
try:
    import yaml
    print(f"  YAML 版本: {yaml.__version__ if hasattr(yaml, '__version__') else '未知'}")
    print("  ✓ YAML 正常")
except Exception as e:
    print(f"  ✗ YAML 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试6: h5py
print("\n[6/10] 测试 h5py...")
try:
    import h5py
    print(f"  h5py 版本: {h5py.__version__}")
    print("  ✓ h5py 正常")
except ImportError:
    print("  ⚠ h5py 未安装（可选）")
except Exception as e:
    print(f"  ✗ h5py 错误: {e}")
    import traceback
    traceback.print_exc()

# 测试7: 加载配置文件
print("\n[7/10] 测试加载配置文件...")
try:
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"  配置文件加载成功")
        print(f"  配置键: {list(config.keys())}")
        print("  ✓ 配置文件正常")
    else:
        print(f"  ⚠ 配置文件不存在: {config_path}")
except Exception as e:
    print(f"  ✗ 配置文件错误: {e}")
    import traceback
    traceback.print_exc()

# 测试8: 导入 sync_data_collector
print("\n[8/10] 测试导入 sync_data_collector...")
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_coll.sync_data_collector import CameraReader, EncoderReader, SensorFrame
    print("  ✓ sync_data_collector 导入正常")
except Exception as e:
    print(f"  ✗ sync_data_collector 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试9: 导入 hand_collector
print("\n[9/10] 测试导入 hand_collector...")
try:
    from hand_collector import HandCollector, HandFrame, visualize_hand
    print("  ✓ dual_hand_collector 导入正常")
except Exception as e:
    print(f"  ✗ dual_hand_collector 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试10: VideoCapture 初始化（可能导致崩溃）
print("\n[10/10] 测试 VideoCapture 初始化...")
try:
    # 尝试创建一个虚拟的 VideoCapture（不连接真实设备）
    cap = cv2.VideoCapture(-1)  # 无效设备号
    is_opened = cap.isOpened()
    cap.release()
    print(f"  VideoCapture 创建测试: opened={is_opened}")
    print("  ✓ VideoCapture 初始化正常")
except Exception as e:
    print(f"  ✗ VideoCapture 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("诊断完成！")
print("=" * 60)
print("\n如果程序运行到这里没有崩溃，说明基础库加载正常。")
print("崩溃可能发生在：")
print("1. VideoCapture 连接真实摄像头时")
print("2. 特定的 OpenCV 函数调用时")
print("3. 角度编码器初始化时")
print("\n建议解决方案：")
print("1. 重新编译安装 OpenCV（不使用AVX2等高级指令集）：")
print("   pip uninstall opencv-python opencv-contrib-python")
print("   pip install opencv-python-headless")
print("\n2. 或者使用 conda 安装（通常有更好的兼容性）：")
print("   conda install -c conda-forge opencv")
print("\n3. 检查 NumPy 版本兼容性：")
print("   pip install numpy==1.23.5")

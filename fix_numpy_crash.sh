#!/bin/bash
# 修复 NumPy "Illegal instruction" 错误

echo "=========================================="
echo "修复 NumPy Illegal Instruction 错误"
echo "=========================================="

# 方案1: 使用 conda 安装不使用 AVX2 的 NumPy（推荐）
echo ""
echo "[方案1] 使用 conda 安装（推荐，最兼容）"
echo "运行以下命令："
echo ""
echo "conda remove -y numpy"
echo "conda install -c conda-forge numpy=1.19.5"
echo ""

# 方案2: 使用 pip 安装旧版本 NumPy
echo "[方案2] 使用 pip 安装旧版本"
echo "运行以下命令："
echo ""
echo "pip uninstall -y numpy"
echo "pip install numpy==1.19.5"
echo ""

# 方案3: 设置环境变量禁用 AVX2
echo "[方案3] 使用环境变量禁用 AVX2（临时方案）"
echo "运行以下命令："
echo ""
echo "export OPENBLAS_CORETYPE=ARMV8"
echo "export OMP_NUM_THREADS=1"
echo "python3 ./host_computer/diagnose_crash.py"
echo ""

echo "=========================================="
echo "推荐步骤："
echo "=========================================="
echo ""
echo "1. 激活conda环境："
echo "   conda activate DataCollectEnv"
echo ""
echo "2. 卸载当前NumPy和OpenCV："
echo "   pip uninstall -y numpy opencv-python opencv-contrib-python"
echo ""
echo "3. 从conda-forge安装兼容版本："
echo "   conda install -c conda-forge numpy=1.19.5 opencv=4.5.1"
echo ""
echo "4. 重新运行诊断脚本："
echo "   python3 ./host_computer/diagnose_crash.py"
echo ""
echo "5. 如果仍有问题，尝试方案3的环境变量方法"
echo ""

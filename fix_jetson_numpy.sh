#!/bin/bash
# Jetson Xavier NumPy/OpenCV Illegal Instruction 修复脚本

echo "=========================================="
echo "Jetson Xavier NumPy/OpenCV 修复"
echo "=========================================="

# 检测平台
if [ ! -f /etc/nv_tegra_release ]; then
    echo "❌ 未检测到 Jetson 平台"
    exit 1
fi

echo "✓ 检测到 Jetson 平台"
echo ""

# 获取 Jetson 版本
jetson_release=$(cat /etc/nv_tegra_release | head -1)
echo "系统信息: $jetson_release"
echo ""

echo "=========================================="
echo "步骤1: 卸载不兼容的 pip 包"
echo "=========================================="
echo ""

# 激活环境
echo "请确保已激活 conda 环境！"
echo ""

pip uninstall -y numpy opencv-python opencv-contrib-python 2>/dev/null || true

echo ""
echo "=========================================="
echo "步骤2: 安装 Jetson 官方系统包"
echo "=========================================="
echo ""

# 方案A: 使用系统 apt 包（推荐）
echo "[方案A] 使用系统 apt 包（推荐）"
echo ""

# 检查并安装依赖
echo "检查 Jetson 官方库..."

sudo apt update -qq
echo "  安装 OpenCV..."
sudo apt install -y python3-opencv 2>/dev/null

echo "  安装 NumPy..."
sudo apt install -y python3-numpy 2>/dev/null

echo ""
echo "验证安装..."

# 验证能否导入（不要直接导入，因为会崩溃）
echo "请运行以下命令验证："
echo ""
echo "  python3 -c \"import cv2; print(cv2.__version__)\""
echo "  python3 -c \"import numpy; print(numpy.__version__)\""
echo ""

echo "=========================================="
echo "步骤3: 如果 apt 包未解决，使用环境变量方案"
echo "=========================================="
echo ""

cat > ~/.bashrc_jetson_fix << 'EOF'
# Jetson Xavier NumPy/OpenCV 修复
export OPENBLAS_CORETYPE=ARMV8
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF

echo "已创建 ~/.bashrc_jetson_fix 文件"
echo ""
echo "每次运行前，执行："
echo "  source ~/.bashrc_jetson_fix"
echo ""

echo "=========================================="
echo "步骤4: 终极方案 - 编译安装"
echo "=========================================="
echo ""

cat > ~/install_numpy_jetson.sh << 'EOF'
#!/bin/bash
# 从源码编译 NumPy（兼容 Jetson ARM64）

echo "从源码编译 NumPy..."

pip install --no-binary :all: numpy==1.19.5 --no-cache-dir -v

echo "从源码编译 OpenCV..."

pip install --no-binary :all: opencv-python==4.5.0.48 --no-cache-dir -v

echo "完成！"
EOF

chmod +x ~/install_numpy_jetson.sh

echo "已创建 ~/install_numpy_jetson.sh 脚本（耗时较长）"
echo "如需使用，运行："
echo "  bash ~/install_numpy_jetson.sh"
echo ""

echo "=========================================="
echo "推荐解决方案顺序"
echo "=========================================="
echo ""
echo "1️⃣  首先尝试 apt 包（刚刚已执行）："
echo "   sudo apt install python3-opencv python3-numpy"
echo ""
echo "2️⃣  验证："
echo "   python3 -c \"import cv2; print('OpenCV:', cv2.__version__)\""
echo "   python3 -c \"import numpy; print('NumPy:', numpy.__version__)\""
echo ""
echo "3️⃣  如果 apt 不行，使用环境变量："
echo "   source ~/.bashrc_jetson_fix"
echo "   python3 -c \"import cv2; print(cv2.__version__)\""
echo ""
echo "4️⃣  如果以上都不行，编译安装（需要1-2小时）："
echo "   bash ~/install_numpy_jetson.sh"
echo ""
echo "=========================================="

#!/bin/bash
# Jetson Xavier 数据采集启动脚本

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Jetson Xavier 数据采集器"
echo "=========================================="
echo ""

# 检查配置文件
CONFIG_FILE="$SCRIPT_DIR/config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误：找不到配置文件 $CONFIG_FILE"
    exit 1
fi

echo "✓ 项目目录: $PROJECT_DIR"
echo "✓ 配置文件: $CONFIG_FILE"
echo ""

# 禁用 GStreamer（使用标准 V4L2）
export USE_GSTREAMER=0
echo "✓ 禁用 GStreamer，使用标准 V4L2 模式"
echo ""

# 选择手和模式
HAND="${1:-right}"
MODE="${2:-visualize}"

if [ "$MODE" = "record" ]; then
    echo "启动模式: 录制"
    echo "手: $HAND"
    echo ""
    echo "按回车开始录制，再按回车停止..."
    echo ""
    python3 "$SCRIPT_DIR/single_hand_collector.py" \
        --config "$CONFIG_FILE" \
        --hand "$HAND" \
        --record
else
    echo "启动模式: 可视化"
    echo "手: $HAND"
    echo "按 'q' 键退出"
    echo ""
    python3 "$SCRIPT_DIR/single_hand_collector.py" \
        --config "$CONFIG_FILE" \
        --hand "$HAND" \
        --visualize
        --realtime-wirte
fi

echo ""
echo "✓ 程序已退出"

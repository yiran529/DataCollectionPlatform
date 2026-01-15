#!/bin/bash
# 创建 udev 规则，永久解决串口权限问题

echo "=================================="
echo "创建串口 udev 规则"
echo "=================================="

# 检查是否为root
if [ "$EUID" -ne 0 ]; then
    echo "请使用 sudo 运行此脚本"
    exit 1
fi

# 检测当前用户（如果通过sudo运行，获取SUDO_USER）
if [ -n "$SUDO_USER" ]; then
    REAL_USER="$SUDO_USER"
else
    REAL_USER="$USER"
fi

echo "用户: $REAL_USER"
echo ""

# 创建 udev 规则文件
RULE_FILE="/etc/udev/rules.d/99-serial-permissions.rules"

cat > "$RULE_FILE" <<EOF
# 串口设备权限规则
# 自动为所有 ttyUSB 设备设置权限

# 为所有 ttyUSB 设备设置权限
KERNEL=="ttyUSB[0-9]*", MODE="0666", GROUP="dialout"

# 如果 dialout 组不存在，使用 users 组
# KERNEL=="ttyUSB[0-9]*", MODE="0666", GROUP="users"
EOF

echo "✓ 已创建 udev 规则文件: $RULE_FILE"
echo ""

# 检查 dialout 组是否存在
if getent group dialout > /dev/null 2>&1; then
    echo "✓ dialout 组存在"
    # 将用户添加到 dialout 组
    usermod -aG dialout "$REAL_USER"
    echo "✓ 已将用户 $REAL_USER 添加到 dialout 组"
else
    echo "⚠️  dialout 组不存在，使用 users 组"
    # 将用户添加到 users 组（通常已存在）
    usermod -aG users "$REAL_USER" 2>/dev/null || true
fi

echo ""
echo "重新加载 udev 规则..."
udevadm control --reload-rules
udevadm trigger

echo ""
echo "=================================="
echo "udev 规则创建完成"
echo "=================================="
echo ""
echo "注意："
echo "1. 需要重新登录或重启才能使组权限生效"
echo "2. 新插入的 USB 串口设备将自动获得正确权限"
echo ""
echo "验证："
echo "  1. 重新登录或重启"
echo "  2. 插入编码器"
echo "  3. 检查权限: ls -l /dev/ttyUSB*"
echo "  4. 应该看到类似: crw-rw-rw- 1 root dialout ..."







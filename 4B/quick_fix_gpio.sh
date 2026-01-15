#!/bin/bash
# 快速修复 GPIO 权限（立即生效）

echo "=================================="
echo "快速修复 GPIO 权限"
echo "=================================="

# 检查是否为root
if [ "$EUID" -ne 0 ]; then
    echo "请使用 sudo 运行此脚本"
    exit 1
fi

# 检测当前用户
if [ -n "$SUDO_USER" ]; then
    REAL_USER="$SUDO_USER"
else
    REAL_USER="$USER"
fi

echo "用户: $REAL_USER"
echo ""

# 1. 确保 gpio 组存在
if ! getent group gpio > /dev/null 2>&1; then
    echo "创建 gpio 组..."
    groupadd gpio
fi

# 2. 添加用户到 gpio 组
echo "添加用户到 gpio 组..."
usermod -aG gpio "$REAL_USER"
echo "✓ 用户已添加到 gpio 组"

# 3. 立即设置 /dev/gpiomem 权限
echo "设置 /dev/gpiomem 权限..."
if [ -e /dev/gpiomem ]; then
    chown root:gpio /dev/gpiomem
    chmod 0660 /dev/gpiomem
    echo "✓ 权限已设置: $(ls -l /dev/gpiomem | awk '{print $1, $3, $4}')"
else
    echo "⚠️ /dev/gpiomem 不存在"
fi

# 4. 创建/更新 udev 规则
echo "创建 udev 规则..."
cat > /etc/udev/rules.d/99-gpio-permissions.rules <<EOF
# GPIO权限规则
SUBSYSTEM=="bcm2835-gpiomem", KERNEL=="gpiomem", GROUP="gpio", MODE="0660"
SUBSYSTEM=="gpio", KERNEL=="gpiochip*", GROUP="gpio", MODE="0660"
EOF

# 5. 重新加载 udev
udevadm control --reload-rules
udevadm trigger

echo ""
echo "=================================="
echo "权限修复完成"
echo "=================================="
echo ""
echo "⚠️  重要：需要重新登录或重启才能使组权限生效"
echo ""
echo "选项1: 重启服务（如果服务以用户运行，需要先重新登录）"
echo "  sudo systemctl restart data_collector"
echo ""
echo "选项2: 使用 root 运行服务（立即生效，推荐）"
echo "  运行: sudo bash install_service.sh"
echo "  当询问 '是否使用root运行? (y/N):' 时输入 y"
echo ""
echo "选项3: 重启系统（使所有权限生效）"
echo "  sudo reboot"
echo ""





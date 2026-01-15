#!/bin/bash
# 修复GPIO权限问题
# 支持树莓派和 Jetson Xavier 平台

set -e

echo "=================================="
echo "修复GPIO权限（树莓派/Jetson Xavier）"
echo "=================================="

# 检查是否为root
if [ "$EUID" -ne 0 ]; then
    echo "请使用 sudo 运行此脚本"
    exit 1
fi

# 检测平台
PLATFORM="unknown"
if [ -f /etc/nv_tegra_release ]; then
    PLATFORM="jetson"
    echo "检测到平台: Nvidia Jetson Xavier"
elif grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    PLATFORM="raspberrypi"
    echo "检测到平台: 树莓派"
else
    echo "检测到平台: 未知（将尝试通用配置）"
fi
echo ""

# 检测当前用户
if [ -n "$SUDO_USER" ]; then
    REAL_USER="$SUDO_USER"
else
    REAL_USER="$USER"
fi

echo "用户: $REAL_USER"
echo ""

# 1. 创建gpio组（如果不存在）
echo "1. 检查gpio组..."
if ! getent group gpio > /dev/null 2>&1; then
    echo "   创建gpio组..."
    groupadd gpio
else
    echo "   gpio组已存在"
fi

# 2. 添加用户到gpio组
echo "2. 添加用户到gpio组..."
usermod -a -G gpio "$REAL_USER"
echo "   用户 $REAL_USER 已添加到gpio组"

# 3. 设置设备权限（根据平台）
echo "3. 设置GPIO设备权限..."
if [ "$PLATFORM" = "jetson" ]; then
    # Jetson 平台使用 /sys/class/gpio
    if [ -d /sys/class/gpio ]; then
        chown -R root:gpio /sys/class/gpio
        chmod -R 0770 /sys/class/gpio
        echo "   Jetson GPIO 权限已设置 (/sys/class/gpio)"
    fi
else
    # 树莓派使用 /dev/gpiomem
    if [ -e /dev/gpiomem ]; then
        chown root:gpio /dev/gpiomem
        chmod 0660 /dev/gpiomem
        echo "   树莓派 gpiomem 权限已设置"
    else
        echo "   ⚠️ /dev/gpiomem 不存在（可能需要重启后才会出现）"
    fi
fi

# 4. 创建udev规则（持久化权限）
echo "4. 创建udev规则..."

# 根据平台创建不同的udev规则
if [ "$PLATFORM" = "jetson" ]; then
    # Jetson 平台的 GPIO 规则
    cat > /etc/udev/rules.d/99-gpio-permissions.rules <<EOF
# Jetson GPIO权限规则
SUBSYSTEM=="gpio", KERNEL=="gpiochip*", GROUP="gpio", MODE="0660"
SUBSYSTEM=="gpio*", PROGRAM="/bin/sh -c 'chown -R root:gpio /sys/class/gpio && chmod -R 770 /sys/class/gpio'"
EOF
else
    # 树莓派的 GPIO 规则
    cat > /etc/udev/rules.d/99-gpio-permissions.rules <<EOF
# 树莓派 GPIO权限规则
SUBSYSTEM=="bcm2835-gpiomem", KERNEL=="gpiomem", GROUP="gpio", MODE="0660"
SUBSYSTEM=="gpio", KERNEL=="gpiochip*", GROUP="gpio", MODE="0660"
EOF
fi

echo "   udev规则已创建"

# 5. 重新加载udev规则
echo "5. 重新加载udev规则..."
udevadm control --reload-rules
udevadm trigger

echo ""
echo "=================================="
echo "权限修复完成!"
echo "=================================="
echo ""
echo "重要：需要重新登录或重启才能生效！"
echo ""
echo "选项1: 重新登录"
echo "  注销并重新登录"
echo ""
echo "选项2: 重启（推荐）"
echo "  sudo reboot"
echo ""
echo "选项3: 临时生效（当前会话）"
echo "  newgrp gpio"
echo "  然后测试: sudo python ../tools/gpio_test/test_led_service.py"
echo ""




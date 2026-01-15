#!/bin/bash
# 修复 /dev/gpiomem 权限

echo "=================================="
echo "修复 /dev/gpiomem 权限"
echo "=================================="

# 检查是否为root
if [ "$EUID" -ne 0 ]; then
    echo "请使用 sudo 运行此脚本"
    exit 1
fi

# 检查设备是否存在
if [ ! -e /dev/gpiomem ]; then
    echo "⚠️  /dev/gpiomem 不存在"
    exit 1
fi

# 检查 gpio 组是否存在
if ! getent group gpio > /dev/null 2>&1; then
    echo "创建 gpio 组..."
    groupadd gpio
fi

# 设置设备权限
echo "设置 /dev/gpiomem 权限..."
chown root:gpio /dev/gpiomem
chmod 0660 /dev/gpiomem
echo "✓ 权限已设置: $(ls -l /dev/gpiomem)"

# 创建 udev 规则（永久解决）
echo ""
echo "创建 udev 规则..."
cat > /etc/udev/rules.d/99-gpiomem-permissions.rules <<EOF
# GPIO 内存设备权限规则
SUBSYSTEM=="bcm2835-gpiomem", KERNEL=="gpiomem", GROUP="gpio", MODE="0660"
SUBSYSTEM=="gpio", KERNEL=="gpiochip*", GROUP="gpio", MODE="0660"
EOF

echo "✓ udev 规则已创建"

# 重新加载 udev 规则
udevadm control --reload-rules
udevadm trigger

echo ""
echo "=================================="
echo "权限修复完成"
echo "=================================="
echo ""
echo "注意："
echo "1. 确保用户 $USER 在 gpio 组中"
echo "2. 如果用户不在 gpio 组中，运行："
echo "   sudo usermod -aG gpio $USER"
echo "3. 然后重新登录或重启"
echo ""




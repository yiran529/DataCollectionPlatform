#!/bin/bash
# 修复串口权限脚本
# 自动检测并设置所有 ttyUSB 设备的权限

echo "=================================="
echo "修复串口权限"
echo "=================================="

# 检查是否为root
if [ "$EUID" -ne 0 ]; then
    echo "请使用 sudo 运行此脚本"
    exit 1
fi

# 查找所有 ttyUSB 设备
devices=$(ls /dev/ttyUSB* 2>/dev/null)

if [ -z "$devices" ]; then
    echo "⚠️  未找到 /dev/ttyUSB* 设备"
    echo "   请确保编码器已连接"
    exit 1
fi

echo "检测到以下串口设备："
for device in $devices; do
    echo "  - $device"
done

echo ""
echo "设置权限..."

# 为每个设备设置权限
for device in $devices; do
    chmod 666 "$device"
    echo "  ✓ $device -> 666"
done

echo ""
echo "=================================="
echo "权限设置完成"
echo "=================================="
echo ""
echo "注意：这些权限在重启后会恢复"
echo "建议创建 udev 规则以永久解决权限问题"
echo ""
echo "运行以下命令创建永久规则："
echo "  sudo bash create_serial_udev_rules.sh"





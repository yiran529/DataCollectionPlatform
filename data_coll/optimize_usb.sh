#!/bin/bash
# USB和相机优化脚本（树莓派）
# 用法: sudo ./optimize_usb.sh

echo "=========================================="
echo "USB 相机优化脚本"
echo "=========================================="

# 检查是否root
if [ "$EUID" -ne 0 ]; then
    echo "请使用 sudo 运行此脚本"
    exit 1
fi

# 1. 增加USB缓冲区
echo ""
echo "[1/5] 增加USB缓冲区..."
CURRENT_USB_MEM=$(cat /sys/module/usbcore/parameters/usbfs_memory_mb 2>/dev/null || echo "16")
echo "  当前: ${CURRENT_USB_MEM} MB"
echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb 2>/dev/null
NEW_USB_MEM=$(cat /sys/module/usbcore/parameters/usbfs_memory_mb 2>/dev/null || echo "N/A")
echo "  设置后: ${NEW_USB_MEM} MB"

# 2. 检查USB设备
echo ""
echo "[2/5] 检查USB设备连接..."
echo "  USB树状图:"
lsusb -t 2>/dev/null | head -20

# 3. 检查视频设备
echo ""
echo "[3/5] 视频设备列表:"
for dev in /dev/video*; do
    if [ -e "$dev" ]; then
        name=$(v4l2-ctl -d "$dev" --info 2>/dev/null | grep "Card type" | cut -d: -f2 | xargs)
        echo "  $dev: $name"
    fi
done

# 4. 检查相机支持的格式
echo ""
echo "[4/5] 检查相机格式 (第一个视频设备)..."
FIRST_DEV=$(ls /dev/video* 2>/dev/null | head -1)
if [ -n "$FIRST_DEV" ]; then
    echo "  设备: $FIRST_DEV"
    v4l2-ctl -d "$FIRST_DEV" --list-formats-ext 2>/dev/null | grep -E "(MJPG|YUYV|H264|Size|Interval)" | head -30
fi

# 5. 设置CPU性能模式
echo ""
echo "[5/5] CPU性能优化..."
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    CURRENT_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    echo "  当前CPU调度: $CURRENT_GOV"
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo "performance" > "$cpu" 2>/dev/null
    done
    NEW_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    echo "  设置后: $NEW_GOV"
else
    echo "  CPU调度不可用"
fi

echo ""
echo "=========================================="
echo "优化建议"
echo "=========================================="
echo ""
echo "1. 确保相机插在 USB 3.0 端口（蓝色）"
echo ""
echo "2. 永久增加USB缓冲区，编辑 /boot/cmdline.txt 添加:"
echo "   usbcore.usbfs_memory_mb=1000"
echo ""
echo "3. 如果使用两个相机，尽量插在不同的USB控制器:"
echo "   - 树莓派4有两个USB控制器"
echo "   - USB2.0端口共享一个控制器"
echo "   - USB3.0端口共享另一个控制器"
echo ""
echo "4. 使用5V 3A以上的电源适配器"
echo ""
echo "5. 如果仍然帧率低，可以尝试降低JPEG质量:"
echo "   v4l2-ctl -d /dev/video0 --set-ctrl compression_quality=80"
echo ""
echo "=========================================="






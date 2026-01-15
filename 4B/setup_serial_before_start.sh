#!/bin/bash
# 服务启动前设置串口权限
for device in /dev/ttyUSB*; do
    if [ -e "$device" ]; then
        chmod 666 "$device" 2>/dev/null || true
    fi
done
exit 0

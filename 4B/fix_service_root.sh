#!/bin/bash
# 修改服务为使用root权限运行

set -e

SERVICE_NAME="data_collector"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "=================================="
echo "修改服务为使用root权限运行"
echo "=================================="

# 检查是否为root
if [ "$EUID" -ne 0 ]; then
    echo "请使用 sudo 运行此脚本:"
    echo "  sudo bash fix_service_root.sh"
    exit 1
fi

# 检查服务文件是否存在
if [ ! -f "$SERVICE_FILE" ]; then
    echo "❌ 服务文件不存在: $SERVICE_FILE"
    echo "   请先运行: sudo bash install_service.sh"
    exit 1
fi

# 备份原文件
cp "$SERVICE_FILE" "${SERVICE_FILE}.backup"
echo "✓ 已备份原服务文件: ${SERVICE_FILE}.backup"

# 读取当前配置
WORK_DIR=$(grep "WorkingDirectory=" "$SERVICE_FILE" | cut -d'=' -f2)
# 解析 ExecStart 行：python路径 脚本路径 --config 配置文件路径 --button 引脚号
EXEC_LINE=$(grep "ExecStart=" "$SERVICE_FILE" | sed 's/ExecStart=//')
PYTHON_PATH=$(echo "$EXEC_LINE" | awk '{print $1}')
SCRIPT_PATH=$(echo "$EXEC_LINE" | awk '{print $2}')
CONFIG_PATH=$(echo "$EXEC_LINE" | awk '{for(i=1;i<=NF;i++) if($i=="--config") print $(i+1)}')
BUTTON_PIN=$(echo "$EXEC_LINE" | awk '{for(i=1;i<=NF;i++) if($i=="--button") print $(i+1)}')
# 如果没有找到按钮引脚，使用默认值18
BUTTON_PIN=${BUTTON_PIN:-18}

# 提取conda环境路径
CONDA_ENV_PATH=$(echo "$PYTHON_PATH" | sed 's|/bin/python3||')
CONDA_ENV=$(basename "$CONDA_ENV_PATH")

echo "检测到配置:"
echo "  工作目录: $WORK_DIR"
echo "  Python路径: $PYTHON_PATH"
echo "  脚本路径: $SCRIPT_PATH"
echo "  配置文件: $CONFIG_PATH"
echo "  按钮引脚: $BUTTON_PIN"
echo "  Conda环境: $CONDA_ENV"
echo ""

# 创建启动前权限设置脚本
SETUP_SCRIPT="$WORK_DIR/setup_serial_before_start.sh"
if [ ! -f "$SETUP_SCRIPT" ]; then
    cat > "$SETUP_SCRIPT" <<'SETUPEOF'
#!/bin/bash
# 服务启动前设置串口权限
for device in /dev/ttyUSB*; do
    if [ -e "$device" ]; then
        chmod 666 "$device" 2>/dev/null || true
    fi
done
exit 0
SETUPEOF
    chmod +x "$SETUP_SCRIPT"
    echo "✓ 已创建启动前脚本: $SETUP_SCRIPT"
fi

# 生成新的服务文件（使用root）
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=GPIO Data Collector Service
After=multi-user.target network-online.target
Wants=network-online.target

[Service]
Type=simple
# 使用root运行（避免GPIO权限问题）
User=root
WorkingDirectory=$WORK_DIR
# 启动前设置串口权限
ExecStartPre=$SETUP_SCRIPT
# 使用 conda 环境中的 Python
ExecStart=$PYTHON_PATH $SCRIPT_PATH --config $CONFIG_PATH --button $BUTTON_PIN
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/data_collector.log
StandardError=append:/var/log/data_collector.log

# 环境变量
Environment="PYTHONUNBUFFERED=1"
Environment="PATH=$CONDA_ENV_PATH/bin:/usr/local/bin:/usr/bin:/bin"
Environment="CONDA_DEFAULT_ENV=$CONDA_ENV"
Environment="CONDA_PREFIX=$CONDA_ENV_PATH"

[Install]
WantedBy=multi-user.target
EOF

echo "✓ 服务文件已更新为使用root权限"

# 重新加载 systemd
echo ""
echo "重新加载 systemd..."
systemctl daemon-reload

echo ""
echo "=================================="
echo "修改完成!"
echo "=================================="
echo ""
echo "现在可以重启服务:"
echo "  sudo systemctl restart $SERVICE_NAME"
echo ""
echo "查看服务状态:"
echo "  sudo systemctl status $SERVICE_NAME"
echo ""
echo "查看日志:"
echo "  tail -f /var/log/data_collector.log"
echo ""


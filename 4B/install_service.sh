#!/bin/bash
# 安装数据收集服务到 systemd

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="data_collector"

echo "=================================="
echo "安装数据收集服务"
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

# 检测工作目录
WORK_DIR="$SCRIPT_DIR"
CONFIG_DIR="$(dirname "$SCRIPT_DIR")/data_coll"

echo "检测到:"
echo "  用户: $REAL_USER"
echo "  工作目录: $WORK_DIR"
echo "  配置目录: $CONFIG_DIR"
echo ""

# 询问按钮引脚（BCM编号）
echo ""
read -p "按钮引脚 (BCM编号，默认18，物理引脚12): " button_pin
button_pin=${button_pin:-18}
if [ "$button_pin" = "18" ] || [ "$button_pin" = "12" ]; then
    echo "   使用GPIO18 (物理引脚12)"
    button_pin=18
fi

# 检测 conda 环境
CONDA_ENV="box_tray"
CONDA_BASE="/home/ztlab/miniconda3"
CONDA_ENV_PATH="$CONDA_BASE/envs/$CONDA_ENV"
PYTHON_PATH="$CONDA_ENV_PATH/bin/python3"

# 检查 conda 环境是否存在
if [ ! -f "$PYTHON_PATH" ]; then
    echo "⚠️  警告: 未找到 conda 环境 $CONDA_ENV"
    echo "   路径: $PYTHON_PATH"
    echo "   将尝试使用系统 Python3"
    PYTHON_PATH="/usr/bin/python3"
else
    echo "✓ 检测到 conda 环境: $CONDA_ENV"
    echo "  Python 路径: $PYTHON_PATH"
fi

# 询问是否使用root运行（如果权限有问题）
echo ""
read -p "如果GPIO权限有问题，是否使用root运行? (y/N): " use_root
use_root=${use_root:-N}

# 创建启动前权限设置脚本
echo "1. 创建启动前权限设置脚本..."
SETUP_SCRIPT="$WORK_DIR/setup_serial_before_start.sh"
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
echo "   ✓ 已创建: $SETUP_SCRIPT"

# 生成服务文件
echo "2. 生成服务文件..."
if [ "$use_root" = "y" ] || [ "$use_root" = "Y" ]; then
    echo "   使用root运行（不推荐，但可以避免权限问题）"
    cat > /etc/systemd/system/${SERVICE_NAME}.service <<EOF
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
ExecStart=$PYTHON_PATH $WORK_DIR/gpio_data_collector.py --config $CONFIG_DIR/config.yaml --button $button_pin
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
else
    echo "   使用用户 $REAL_USER 运行"
    cat > /etc/systemd/system/${SERVICE_NAME}.service <<EOF
[Unit]
Description=GPIO Data Collector Service
After=multi-user.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$REAL_USER
WorkingDirectory=$WORK_DIR
# 使用 conda 环境中的 Python
ExecStart=$PYTHON_PATH $WORK_DIR/gpio_data_collector.py --config $CONFIG_DIR/config.yaml --button $button_pin
Restart=on-failure
RestartSec=5
StandardOutput=append:/var/log/data_collector.log
StandardError=append:/var/log/data_collector.log

# 环境变量
Environment="PYTHONUNBUFFERED=1"
Environment="PATH=$CONDA_ENV_PATH/bin:/usr/local/bin:/usr/bin:/bin"
Environment="CONDA_DEFAULT_ENV=$CONDA_ENV"
Environment="CONDA_PREFIX=$CONDA_ENV_PATH"

# 确保GPIO和串口可访问
# 注意：串口权限通过 udev 规则设置，用户需要在 dialout 组中
SupplementaryGroups=gpio dialout

[Install]
WantedBy=multi-user.target
EOF
fi

# 创建日志文件
echo "3. 创建日志文件..."
touch /var/log/data_collector.log
chown ${REAL_USER}:${REAL_USER} /var/log/data_collector.log

# 询问是否创建串口权限 udev 规则（永久解决）
echo ""
read -p "是否创建串口权限 udev 规则（永久解决权限问题）? (Y/n): " create_udev
create_udev=${create_udev:-Y}
if [ "$create_udev" = "Y" ] || [ "$create_udev" = "y" ]; then
    echo "   创建 udev 规则..."
    bash "$WORK_DIR/create_serial_udev_rules.sh"
    echo "   ✓ udev 规则已创建"
    echo "   注意：需要重新登录或重启才能使组权限生效"
fi

# 重新加载 systemd
echo "4. 重新加载 systemd..."
systemctl daemon-reload

# 启用服务
echo "5. 启用开机自启..."
systemctl enable $SERVICE_NAME

echo ""
echo "=================================="
echo "安装完成!"
echo "=================================="
echo ""
echo "常用命令:"
echo "  启动服务:    sudo systemctl start $SERVICE_NAME"
echo "  停止服务:    sudo systemctl stop $SERVICE_NAME"
echo "  重启服务:    sudo systemctl restart $SERVICE_NAME"
echo "  查看状态:    sudo systemctl status $SERVICE_NAME"
echo "  查看日志:    tail -f /var/log/data_collector.log"
echo "  禁用自启:    sudo systemctl disable $SERVICE_NAME"
echo ""
echo "服务将在下次开机时自动启动"
echo "现在可以手动启动: sudo systemctl start $SERVICE_NAME"
echo ""


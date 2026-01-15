# 立即修复 GPIO 权限问题

## 当前问题

`/dev/gpiomem` 权限为 `600`（只有 root 可以访问），需要改为 `660`（gpio 组可访问）。

## 立即修复（两个步骤）

### 步骤1：修复设备权限

```bash
sudo chown root:gpio /dev/gpiomem
sudo chmod 0660 /dev/gpiomem
```

### 步骤2：创建永久规则（防止重启后恢复）

```bash
sudo bash /home/ztlab/DataCollectionPlatform/4B/quick_fix_gpio.sh
```

或者手动创建 udev 规则：

```bash
sudo bash -c 'cat > /etc/udev/rules.d/99-gpio-permissions.rules <<EOF
SUBSYSTEM=="bcm2835-gpiomem", KERNEL=="gpiomem", GROUP="gpio", MODE="0660"
SUBSYSTEM=="gpio", KERNEL=="gpiochip*", GROUP="gpio", MODE="0660"
EOF'

sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 步骤3：重启服务

```bash
sudo systemctl restart data_collector
tail -f /var/log/data_collector.log
```

## 或者：使用 root 运行服务（最快）

如果权限修复仍然有问题，可以使用 root 运行服务：

```bash
cd /home/ztlab/DataCollectionPlatform/4B

# 停止现有服务
sudo systemctl stop data_collector
sudo systemctl disable data_collector

# 重新安装，选择使用 root
sudo bash install_service.sh
# 当询问 "是否使用root运行? (y/N):" 时输入 y

# 启动服务
sudo systemctl start data_collector
```

## 验证

```bash
# 检查权限
ls -l /dev/gpiomem
# 应该显示: crw-rw---- 1 root gpio ... (0660)

# 检查服务
sudo systemctl status data_collector
tail -f /var/log/data_collector.log
```





# GPIO权限问题修复指南

## 问题症状

运行服务时出现错误：
```
no access to /dev/mem try running as root
```

## 解决方案

### 方案1：修复GPIO权限（推荐）

运行权限修复脚本：

```bash
cd /home/你的用户名/Documents/DataCollectionPlatform/4B
sudo bash fix_gpio_permissions.sh
```

然后**重新登录**或**重启**：

```bash
sudo reboot
```

### 方案2：使用root运行服务（快速但不推荐）

在安装服务时选择使用root运行：

```bash
# 安装测试服务
sudo bash install_test_service.sh
# 当询问 "是否使用root运行? (y/N):" 时输入 y

# 或安装数据收集服务
sudo bash install_service.sh
# 当询问 "是否使用root运行? (y/N):" 时输入 y
```

### 方案3：手动修复权限

```bash
# 1. 创建gpio组（如果不存在）
sudo groupadd gpio

# 2. 添加用户到gpio组
sudo usermod -a -G gpio $USER

# 3. 设置/dev/gpiomem权限
sudo chown root:gpio /dev/gpiomem
sudo chmod 0660 /dev/gpiomem

# 4. 创建udev规则（持久化）
sudo bash -c 'cat > /etc/udev/rules.d/99-gpio-permissions.rules <<EOF
SUBSYSTEM=="bcm2835-gpiomem", KERNEL=="gpiomem", GROUP="gpio", MODE="0660"
SUBSYSTEM=="gpio", KERNEL=="gpiochip*", GROUP="gpio", MODE="0660"
EOF'

# 5. 重新加载udev规则
sudo udevadm control --reload-rules
sudo udevadm trigger

# 6. 重新登录或重启
sudo reboot
```

## 验证修复

### 方法1：检查用户组

```bash
groups
# 应该看到 gpio 在列表中
```

### 方法2：检查/dev/gpiomem权限

```bash
ls -l /dev/gpiomem
# 应该显示类似: crw-rw---- 1 root gpio ...
```

### 方法3：手动测试

```bash
# 临时切换到gpio组
newgrp gpio

# 测试程序
sudo python ../tools/gpio_test/test_led_service.py
```

## 如果仍然有问题

### 检查服务配置

```bash
# 查看服务文件
cat /etc/systemd/system/test_led.service

# 确保有 SupplementaryGroups=gpio
# 或者使用 User=root
```

### 查看详细日志

```bash
# 查看服务日志
sudo journalctl -u test_led -n 50

# 查看系统日志
sudo dmesg | grep -i gpio
```

### 重新安装服务（使用root）

如果权限修复不工作，可以重新安装服务并选择使用root：

```bash
# 停止并删除旧服务
sudo systemctl stop test_led
sudo systemctl disable test_led
sudo rm /etc/systemd/system/test_led.service
sudo systemctl daemon-reload

# 重新安装（选择使用root）
sudo bash install_test_service.sh
# 输入 y 使用root运行
```

## 推荐方案

**最佳实践**：
1. 先运行 `fix_gpio_permissions.sh` 修复权限
2. 重新登录或重启
3. 使用普通用户运行服务（更安全）

**快速方案**：
- 如果权限修复不工作，临时使用root运行服务
- 注意：使用root运行有安全风险，但可以快速解决问题

## 常见问题

### Q: 为什么需要重新登录？

A: 用户组更改需要重新登录才能生效。`newgrp gpio` 可以临时生效，但重启后会丢失。

### Q: 使用root运行安全吗？

A: 不推荐，但有GPIO权限问题时的快速解决方案。如果只是测试，可以临时使用。

### Q: 修复后还是不行？

A: 检查：
1. 用户是否真的在gpio组中：`groups`
2. /dev/gpiomem权限是否正确：`ls -l /dev/gpiomem`
3. udev规则是否生效：`udevadm test /sys/class/gpio`




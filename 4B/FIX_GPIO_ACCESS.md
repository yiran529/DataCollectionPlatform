# 修复 GPIO 访问权限问题

## 问题

服务启动时出现错误：
```
RuntimeError: No access to /dev/mem. Try running as root!
```

这是因为 `/dev/gpiomem` 的权限不正确，普通用户无法访问。

## 解决方案

### 方案1：修复 GPIO 权限（推荐）

运行权限修复脚本：

```bash
cd /home/ztlab/DataCollectionPlatform/4B
sudo bash fix_gpio_permissions.sh
```

这个脚本会：
1. 确保用户在 `gpio` 组中
2. 设置 `/dev/gpiomem` 权限为 `0660`（gpio 组可读写）
3. 创建 udev 规则，永久解决权限问题

**重要**：修复后需要**重新登录**或**重启**才能生效。

### 方案2：使用 root 运行服务（快速但不推荐）

重新安装服务，选择使用 root 运行：

```bash
cd /home/ztlab/DataCollectionPlatform/4B

# 停止现有服务
sudo systemctl stop data_collector
sudo systemctl disable data_collector

# 重新安装，当询问 "是否使用root运行? (y/N):" 时输入 y
sudo bash install_service.sh

# 启动服务
sudo systemctl start data_collector
```

## 验证修复

### 检查权限

```bash
# 检查 /dev/gpiomem 权限
ls -l /dev/gpiomem
# 应该显示: crw-rw---- 1 root gpio ... (0660)

# 检查用户组
groups
# 应该看到 gpio 在列表中
```

### 测试服务

```bash
# 重启服务
sudo systemctl restart data_collector

# 查看日志
tail -f /var/log/data_collector.log
# 应该不再出现权限错误
```

## 如果仍然有问题

### 临时解决方案（当前会话）

```bash
# 切换到 gpio 组
newgrp gpio

# 然后手动测试
python3 /home/ztlab/DataCollectionPlatform/4B/gpio_data_collector.py
```

### 检查服务配置

```bash
# 查看服务配置
cat /etc/systemd/system/data_collector.service | grep -E "User=|SupplementaryGroups="

# 确保有 SupplementaryGroups=gpio
```

## 推荐流程

1. **运行权限修复脚本**：
   ```bash
   sudo bash fix_gpio_permissions.sh
   ```

2. **重新登录或重启**：
   ```bash
   sudo reboot
   ```

3. **验证权限**：
   ```bash
   ls -l /dev/gpiomem
   groups
   ```

4. **重启服务**：
   ```bash
   sudo systemctl restart data_collector
   tail -f /var/log/data_collector.log
   ```






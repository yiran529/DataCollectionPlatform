# 串口权限问题解决方案

## 问题描述

编码器通过 USB 转串口连接（如 `/dev/ttyUSB0`），经常遇到权限问题：
- 错误：`Permission denied: '/dev/ttyUSB0'`
- 需要手动执行：`sudo chmod 777 /dev/ttyUSB0`
- 串口号可能变化：`/dev/ttyUSB0` → `/dev/ttyUSB1`

## 解决方案

### 方案1：创建 udev 规则（推荐，永久解决）

这是**最推荐的方案**，可以永久解决权限问题：

```bash
cd /home/ztlab/DataCollectionPlatform/4B
sudo bash create_serial_udev_rules.sh
```

这个脚本会：
1. 创建 udev 规则，自动为所有 `ttyUSB*` 设备设置权限
2. 将用户添加到 `dialout` 组
3. 重新加载 udev 规则

**注意**：需要重新登录或重启才能使组权限生效。

### 方案2：临时设置权限（快速但不持久）

如果需要临时解决权限问题：

```bash
sudo bash fix_serial_permissions.sh
```

这个脚本会为当前所有 `ttyUSB*` 设备设置权限，但重启后会恢复。

### 方案3：使用 root 运行服务（简单但不推荐）

在安装服务时选择使用 root 运行：

```bash
sudo bash install_service.sh
# 当询问 "是否使用root运行? (y/N):" 时输入 y
```

这样可以避免权限问题，但不推荐从安全角度考虑。

## 自动串口检测

程序已经支持**自动检测串口设备**：

- 如果配置文件中指定的串口不存在，会自动查找第一个可用的 `ttyUSB*` 设备
- 代码位置：`data_coll/sync_data_collector.py` 第 425-426 行

```python
ports = sorted(glob.glob('/dev/ttyUSB*'))
port = cfg.encoder_port if cfg.encoder_port in ports else (ports[0] if ports else None)
```

这意味着：
- 如果 `/dev/ttyUSB0` 不存在，会自动使用 `/dev/ttyUSB1`
- 如果配置文件中指定了 `/dev/ttyUSB0`，但实际是 `/dev/ttyUSB1`，会自动切换到 `/dev/ttyUSB1`

## 服务启动时的权限处理

服务安装脚本已经配置为：
1. 在安装时询问是否创建 udev 规则
2. 将用户添加到 `dialout` 组
3. 服务配置中包含 `SupplementaryGroups=dialout`

## 验证权限

### 检查用户组

```bash
groups
# 应该看到 dialout 在列表中
```

### 检查设备权限

```bash
ls -l /dev/ttyUSB*
# 应该显示类似: crw-rw-rw- 1 root dialout ...
```

### 检查 udev 规则

```bash
cat /etc/udev/rules.d/99-serial-permissions.rules
```

## 故障排查

### 问题1：仍然提示权限错误

1. **检查用户组**：
   ```bash
   groups
   # 如果没有 dialout，需要重新登录
   ```

2. **检查 udev 规则**：
   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

3. **手动设置权限**（临时）：
   ```bash
   sudo chmod 666 /dev/ttyUSB*
   ```

### 问题2：串口号变化

程序已经支持自动检测，但如果需要固定串口号：

1. **使用 udev 规则固定设备名称**：
   ```bash
   # 查看设备信息
   udevadm info -a -n /dev/ttyUSB0 | grep -E "ATTRS\{idVendor\}|ATTRS\{idProduct\}|ATTRS\{serial\}"
   
   # 创建固定名称规则（示例）
   sudo nano /etc/udev/rules.d/99-encoder-serial.rules
   # 添加规则，例如：
   # SUBSYSTEM=="tty", ATTRS{idVendor}=="1234", ATTRS{idProduct}=="5678", SYMLINK+="encoder", MODE="0666"
   ```

2. **修改配置文件**：
   在 `data_coll/config.yaml` 中指定固定设备名称（如果使用 udev 规则创建了固定名称）

## 完整安装流程

1. **安装服务**：
   ```bash
   sudo bash install_service.sh
   ```

2. **创建 udev 规则**（如果安装时没有创建）：
   ```bash
   sudo bash create_serial_udev_rules.sh
   ```

3. **重新登录或重启**：
   ```bash
   sudo reboot
   ```

4. **验证**：
   ```bash
   # 检查权限
   ls -l /dev/ttyUSB*
   
   # 启动服务
   sudo systemctl start data_collector
   
   # 查看日志
   tail -f /var/log/data_collector.log
   ```

## 相关文件

- `fix_serial_permissions.sh` - 临时设置权限脚本
- `create_serial_udev_rules.sh` - 创建永久 udev 规则脚本
- `install_service.sh` - 服务安装脚本（已集成权限处理）


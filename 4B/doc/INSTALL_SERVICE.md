# 开机自启动安装指南

## 快速安装

```bash
cd /home/你的用户名/Documents/DataCollectionPlatform/4B
sudo bash install_service.sh
```

## 安装步骤详解

### 1. 运行安装脚本

```bash
sudo bash install_service.sh
```

脚本会自动：
- 检测当前用户
- 检测工作目录
- 生成 systemd 服务文件
- 创建日志文件
- 启用开机自启动

### 2. 启动服务（可选，测试）

```bash
sudo systemctl start data_collector
```

### 3. 查看服务状态

```bash
sudo systemctl status data_collector
```

### 4. 查看日志

```bash
tail -f /var/log/data_collector.log
```

## 常用命令

| 操作 | 命令 |
|------|------|
| 启动服务 | `sudo systemctl start data_collector` |
| 停止服务 | `sudo systemctl stop data_collector` |
| 重启服务 | `sudo systemctl restart data_collector` |
| 查看状态 | `sudo systemctl status data_collector` |
| 查看日志 | `tail -f /var/log/data_collector.log` |
| 禁用自启 | `sudo systemctl disable data_collector` |
| 启用自启 | `sudo systemctl enable data_collector` |

## 验证开机自启

1. 重启树莓派：
   ```bash
   sudo reboot
   ```

2. 重启后检查服务状态：
   ```bash
   sudo systemctl status data_collector
   ```

3. 应该看到服务正在运行（`active (running)`）

## 故障排查

### 服务无法启动

1. 查看详细日志：
   ```bash
   sudo journalctl -u data_collector -n 50
   ```

2. 检查配置文件路径：
   ```bash
   cat /etc/systemd/system/data_collector.service
   ```

3. 手动测试程序：
   ```bash
   cd /home/你的用户名/Documents/DataCollectionPlatform/4B
   python3 gpio_data_collector.py --keyboard
   ```

### GPIO权限问题

如果出现GPIO权限错误：

```bash
# 添加用户到gpio组
sudo usermod -aG gpio $USER

# 重新登录生效
# 或者重启
sudo reboot
```

### 修改服务配置

如果需要修改服务配置：

1. 编辑服务文件：
   ```bash
   sudo nano /etc/systemd/system/data_collector.service
   ```

2. 重新加载并重启：
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart data_collector
   ```

## 卸载服务

```bash
sudo systemctl stop data_collector
sudo systemctl disable data_collector
sudo rm /etc/systemd/system/data_collector.service
sudo systemctl daemon-reload
```

## ⚠️ 重要说明

### 键盘监听在服务模式下不可用

**当程序作为 systemd 服务运行时，无法监听键盘输入**（服务没有标准输入）。

**解决方案：使用GPIO按钮**

服务已配置为使用GPIO按钮（物理按钮），这是最可靠的方式：

- **按钮接线**：
  - 一端 → GPIO18 (物理引脚12)
  - 另一端 → GND (物理引脚6/9/14等)

- **操作方式**：
  - 按下按钮 → 开始/停止录制
  - LED状态指示：
    - 🔴 红色：等待录制
    - 🔵 蓝色闪烁：初始化中
    - 🟢 绿色：正在录制
    - 🟢 绿色快闪：保存中

如果需要使用键盘，请**不要使用服务模式**，直接在终端运行：
```bash
sudo python gpio_data_collector.py --keyboard
```

详见 `SERVICE_MODE.md` 了解更多详情。


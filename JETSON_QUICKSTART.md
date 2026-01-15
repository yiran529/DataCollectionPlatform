# Jetson Xavier 快速开始

这是一个简化的快速开始指南，帮助您在 Jetson Xavier 上快速运行本项目。

## 前置条件

- ✅ Nvidia Jetson Xavier 开发板
- ✅ JetPack 4.6+ 已安装
- ✅ Python 3.6+
- ✅ 硬件连接完成（GPIO 按钮、LED、摄像头）

## 5分钟快速启动

### 1. 克隆项目

```bash
cd ~/projects
git clone <repository-url> DataCollectionPlatform
cd DataCollectionPlatform
```

### 2. 安装依赖

```bash
# 安装 Jetson.GPIO（如未安装）
sudo pip3 install Jetson.GPIO

# 或使用 apt（推荐）
sudo apt install python3-jetson-gpio

# 安装其他 Python 依赖
pip3 install opencv-python numpy h5py minimalmodbus pyyaml

# 验证安装
python3 -c "import Jetson.GPIO; print('✓ Jetson.GPIO 可用')"
```

### 3. 配置 GPIO 权限

```bash
cd 4B
sudo bash fix_gpio_permissions.sh
```

**重要：** 运行后需要重新登录以使权限生效。

```bash
# 重新登录
logout
# 或重启
sudo reboot
```

### 4. 测试 GPIO

```bash
cd 4B

# 测试 LED 和按钮
sudo python3 test_gpio_led.py

# 或测试按钮单独功能
sudo python3 test_button.py
```

**预期输出：**
- LED 会依次点亮红、绿、蓝色
- 按下按钮时会有响应

### 5. 测试摄像头

```bash
cd ../data_coll

# 列出可用摄像头
v4l2-ctl --list-devices

# 测试摄像头帧率
python3 test_camera_fps.py
```

### 6. 运行数据采集

```bash
cd ../4B

# 使用配置文件运行
python3 gpio_data_collector.py --config ../data_coll/config.yaml

# 或指定自定义按钮引脚
python3 gpio_data_collector.py --config ../data_coll/config.yaml --button 18
```

**操作说明：**
1. 程序启动后 LED 显示红色（等待）
2. 按下按钮 → 蓝色闪烁（初始化）
3. 初始化完成 → 绿色（开始录制）
4. 再次按下 → 红色（停止并保存）

### 7. 查看数据

```bash
# 数据保存在 U 盘或指定目录
ls /media/*/data_*

# 或在项目目录
ls ../data_coll/data_*
```

## 常见问题快速解决

### GPIO 权限错误

```bash
# 症状：RuntimeError: No access to /sys/class/gpio/export
# 解决：
sudo bash 4B/fix_gpio_permissions.sh
logout  # 重新登录
```

### 摄像头无法打开

```bash
# 检查设备
ls /dev/video*

# 添加视频组权限
sudo usermod -aG video $USER
logout
```

### LED 不亮

```bash
# 检查接线
# 确认使用 BCM 引脚编号：
# 红色：GPIO22 (物理引脚15)
# 绿色：GPIO27 (物理引脚13)  
# 蓝色：GPIO23 (物理引脚16)
```

### 性能不佳

```bash
# 切换到最高性能模式
sudo nvpmodel -m 0
sudo jetson_clocks

# 验证
sudo nvpmodel -q
```

## 进阶配置

### 启用开机自启动

```bash
cd 4B
sudo bash install_service.sh

# 启动服务
sudo systemctl start data_collector

# 查看状态
sudo systemctl status data_collector
```

### 自定义配置

编辑配置文件：

```bash
nano data_coll/config.yaml
```

修改摄像头参数：
```yaml
stereo:
  device: 6
  width: 3840
  height: 1080
  fps: 30  # Jetson 可支持更高帧率

mono:
  device: 4
  width: 1280
  height: 1024
  fps: 30
```

## 性能监控

### 实时监控

```bash
# 终端1：运行程序
cd 4B
python3 gpio_data_collector.py --config ../data_coll/config.yaml

# 终端2：监控性能
tegrastats

# 或使用 jtop（需要安装）
sudo pip3 install jetson-stats
sudo jtop
```

### 查看日志

```bash
# 如果使用 systemd 服务
sudo journalctl -u data_collector -f

# 或查看日志文件
tail -f /var/log/data_collector.log
```

## 下一步

- 📖 阅读完整的 [Jetson Xavier 适配指南](./JETSON_XAVIER_GUIDE.md)
- 🔧 查看 [平台适配说明](./PLATFORM_ADAPTATION.md)
- 📋 参考 [GPIO 接线文档](./4B/README.md)

## 需要帮助？

1. 检查 [常见问题](./JETSON_XAVIER_GUIDE.md#常见问题)
2. 查看 [Jetson GPIO 文档](https://github.com/NVIDIA/jetson-gpio)
3. 提交 Issue（包含错误日志和系统信息）

---

**系统信息（用于调试）：**
```bash
# Jetson 版本
cat /etc/nv_tegra_release

# Python 版本
python3 --version

# GPIO 库版本
python3 -c "import Jetson.GPIO; print(Jetson.GPIO.VERSION)"

# OpenCV 版本
python3 -c "import cv2; print(cv2.__version__)"
```

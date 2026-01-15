# 4B 文件夹结构说明

## 主程序文件

### 核心程序
- **gpio_data_collector.py** - GPIO数据收集控制器主程序
  - 通过GPIO按钮控制数据收集
  - RGB LED状态指示
  - 支持开机自启动服务

## 服务安装和管理

### 安装脚本
- **install_service.sh** - 主服务安装脚本
  - 自动创建systemd服务
  - 配置开机自启动
  - 处理GPIO和串口权限

### 权限管理脚本
- **create_serial_udev_rules.sh** - 创建串口权限udev规则（永久解决）
- **fix_serial_permissions.sh** - 临时修复串口权限
- **fix_gpio_permissions.sh** - 修复GPIO权限
- **setup_serial_before_start.sh** - 服务启动前设置串口权限

### 服务文件
- **data_collector.service** - systemd服务配置文件（由install_service.sh自动生成）

## 文档

- **README.md** - 主程序使用说明
- **INSTALL_SERVICE.md** - 服务安装指南
- **SERVICE_MODE.md** - 服务模式说明
- **SERIAL_PERMISSIONS.md** - 串口权限问题解决方案
- **FIX_GPIO_PERMISSIONS.md** - GPIO权限问题修复指南

## 数据文件夹

- **data/** - 收集的数据文件存储目录

## 测试工具位置

所有测试工具已移动到 `tools/gpio_test/` 文件夹：

- test_led_button.py - LED和按钮测试程序
- test_led_service.py - LED服务测试程序
- test_led.service - LED测试服务配置
- install_test_service.sh - 测试服务安装脚本
- uninstall_test_service.sh - 测试服务卸载脚本
- TEST_LED_README.md - 测试工具说明

## 快速开始

### 1. 安装服务

```bash
cd /home/ztlab/DataCollectionPlatform/4B
sudo bash install_service.sh
```

### 2. 启动服务

```bash
sudo systemctl start data_collector
```

### 3. 查看状态

```bash
sudo systemctl status data_collector
tail -f /var/log/data_collector.log
```

## 文件组织原则

- **4B文件夹**：只保留主程序和服务器相关的核心文件
- **tools/gpio_test文件夹**：存放所有测试和调试工具
- 清晰的职责分离，便于维护和管理







# GPIO 数据收集控制器

通过GPIO按钮和RGB LED控制数据收集的开始和结束。

**支持平台：**
- 🥧 树莓派 4B (使用 RPi.GPIO)
- 🚀 Nvidia Jetson Xavier (使用 Jetson.GPIO)

## 硬件连接

### GPIO引脚 (BCM编号)

**注意：** 以下为树莓派的引脚编号。Jetson Xavier 的引脚布局不同，请参考 [Jetson GPIO 文档](https://github.com/NVIDIA/jetson-gpio)。

| 组件 | GPIO | 物理引脚 | 说明 |
|------|------|----------|------|
| 按钮 | GPIO18 | 12 | 另一端接GND |
| LED红 | GPIO22 | 15 | 通过电阻接LED红色引脚 |
| LED绿 | GPIO27 | 13 | 通过电阻接LED绿色引脚 |
| LED蓝 | GPIO23 | 16 | 通过电阻接LED蓝色引脚 |
| GND | - | 6, 9, 14, 20, 25, 30, 34, 39 | 按钮和LED共阴极 |

### 接线图（树莓派）

```
树莓派 GPIO 引脚布局 (从上往下看，USB口朝下)

        3V3  (1)  (2)  5V
      GPIO2  (3)  (4)  5V
      GPIO3  (5)  (6)  GND  ← 按钮和LED的GND
      GPIO4  (7)  (8)  GPIO14
        GND  (9)  (10) GPIO15
按钮 → GPIO18 (11) (12) GPIO18
LED绿 → GPIO27 (13) (14) GND
LED红 → GPIO22 (15) (16) GPIO23 ← LED蓝
      GPIO10 (17) (18) GPIO24
       ...
```

**Jetson Xavier 用户：** 请参考 Jetson 40-pin 扩展接头的引脚定义，使用相同的 BCM 编号。

### 按钮接线
```
GPIO18 ──┬── 按钮 ──── GND
         │
       (内部上拉电阻)
```

### RGB LED接线 (共阴极)
```
GPIO22 ── 330Ω ── LED红色引脚
GPIO27 ── 330Ω ── LED绿色引脚  
GPIO23 ── 330Ω ── LED蓝色引脚
                  LED共阴极 ── GND
```

## 状态指示

| 颜色 | 状态 | 说明 |
|------|------|------|
| 🔴 红色 | IDLE | 等待开始/录制结束 |
| 🔵 蓝色 | INIT | 初始化和预热中 |
| 🟢 绿色 | RECORDING | 正在录制 |

## 操作流程

```
1. 程序启动 → 红色 (等待)
          ↓
2. 按下按钮 → 蓝色 (初始化中)
          ↓
3. 初始化完成 → 绿色 (自动开始录制)
          ↓
4. 按下按钮 → 红色 (停止录制，保存数据)
          ↓
5. 按下按钮 → 绿色 (直接开始录制)
          ↓
   重复 4-5...
```

## 安装和使用

### 1. 安装依赖

**树莓派：**
```bash
pip install RPi.GPIO
```

**Jetson Xavier：**
```bash
# Jetson.GPIO 需要 Python 3.6+ 和 root 权限（或正确的用户组权限）
pip install Jetson.GPIO
```

### 2. 测试运行
```bash
cd /home/cjq/Documents/DataCollectionPlatform/4B
python gpio_data_collector.py --config ../data_coll/config.yaml
```

### 3. 设置开机自启动
```bash
sudo bash install_service.sh
```

### 4. 服务管理
```bash
# 查看状态
sudo systemctl status data_collector

# 启动/停止/重启
sudo systemctl start data_collector
sudo systemctl stop data_collector
sudo systemctl restart data_collector

# 查看日志
tail -f /var/log/data_collector.log
```

## 命令行参数

```
--config, -c    配置文件路径 (默认: ../data_coll/config.yaml)
--button        按钮GPIO引脚 (默认: 18)
--led-red       红色LED GPIO引脚 (默认: 22)
--led-green     绿色LED GPIO引脚 (默认: 27)
--led-blue      蓝色LED GPIO引脚 (默认: 23)
```

## 调试步骤

### 第一步：测试GPIO连接
```bash
cd /home/ztlab/DataCollectionPlatform/tools/gpio_test
sudo python3 test_led_button.py
```

测试程序提供以下功能：
1. **LED单色测试** - 分别点亮红/绿/蓝，确认每个颜色工作正常
2. **LED组合颜色测试** - 测试黄/紫/青/白等组合色
3. **LED闪烁测试** - 红→绿→蓝循环闪烁
4. **按钮测试** - 按5次按钮，每次LED变色
5. **按钮状态监测** - 持续显示按钮状态

### 第二步：测试数据收集（模拟模式）
```bash
# 不接硬件时会进入模拟模式
python gpio_data_collector.py --config ../data_coll/config.yaml
# 按回车键模拟按钮
```

### 第三步：完整测试
```bash
# 接好硬件后完整测试
python gpio_data_collector.py --config ../data_coll/config.yaml
```

### 常见问题排查

| 问题 | 可能原因 | 解决方法 |
|------|----------|----------|
| LED不亮 | 接线错误 | 检查物理引脚编号 |
| 颜色错误 | R/G/B接反 | 对照引脚重新接线 |
| 按钮无响应 | 没接GND | 按钮另一端接GND |
| 权限错误 | 用户不在gpio组 | `sudo usermod -aG gpio $USER` 然后重新登录 |

## 注意事项

1. **电阻**: RGB LED需要串联330Ω-1kΩ限流电阻
2. **共阴/共阳**: 本程序假设使用共阴极LED，如果是共阳极需要修改代码逻辑
3. **GPIO权限**: 确保用户在gpio组中: `sudo usermod -aG gpio $USER`
4. **防抖**: 程序内置300ms按钮防抖


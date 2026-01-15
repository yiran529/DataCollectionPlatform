# 服务模式说明

## ⚠️ 重要：键盘监听在服务模式下不可用

当程序作为 systemd 服务运行时，**无法监听键盘输入**，因为服务没有标准输入（stdin）。

## 解决方案

### ✅ 方案1：使用GPIO按钮（推荐）

**这是最可靠的方式**，程序默认使用GPIO按钮。

#### 接线
- 按钮一端 → GPIO17 (物理引脚11)
- 按钮另一端 → GND (物理引脚6/9/14等)

#### 服务配置
服务已经配置为使用GPIO按钮（不包含 `--keyboard` 参数）。

#### 操作
- **按下按钮** → 开始/停止录制
- LED状态指示：
  - 🔴 红色：等待录制
  - 🔵 蓝色闪烁：初始化中
  - 🟢 绿色：正在录制
  - 🟢 绿色快闪：保存中

---

### 方案2：使用文件触发（如果需要远程控制）

如果你需要通过文件来触发，可以添加文件监控功能。

#### 创建触发文件
```bash
# 开始录制
touch /tmp/data_collector_start

# 停止录制
touch /tmp/data_collector_stop
```

#### 修改程序
需要添加文件监控功能（当前版本不支持）。

---

### 方案3：使用信号（高级）

可以通过发送信号来控制：

```bash
# 发送自定义信号（需要修改程序支持）
kill -USR1 $(pgrep -f gpio_data_collector)
```

---

## 当前服务配置

服务文件 (`/etc/systemd/system/data_collector.service`) 配置为：

```ini
ExecStart=/usr/bin/python3 .../gpio_data_collector.py --config .../config.yaml
```

**注意**：没有 `--keyboard` 参数，所以使用GPIO按钮。

---

## 测试GPIO按钮

如果按钮不工作，可以运行测试程序：

```bash
cd /home/你的用户名/Documents/DataCollectionPlatform/4B
sudo python test_gpio.py
```

选择"测试4: 按钮测试"来验证按钮是否正常工作。

---

## 如果必须使用键盘

如果你**必须**使用键盘控制，可以：

1. **不使用服务模式**，直接在终端运行：
   ```bash
   cd /home/你的用户名/Documents/DataCollectionPlatform/4B
   sudo python gpio_data_collector.py --keyboard
   ```

2. **使用 screen/tmux** 保持会话：
   ```bash
   screen -S datacollector
   sudo python gpio_data_collector.py --keyboard
   # 按 Ctrl+A+D 分离会话
   ```

3. **使用 nohup**：
   ```bash
   nohup sudo python gpio_data_collector.py --keyboard > /tmp/datacollector.log 2>&1 &
   ```

但**推荐使用GPIO按钮**，因为：
- ✅ 更可靠（不依赖终端）
- ✅ 适合嵌入式应用
- ✅ 开机自启动后即可使用
- ✅ 无需SSH连接




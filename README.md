# DataCollectionPlatform 核心结构与运行入口

本文档覆盖项目中**最核心**的采集相关文件，其它文件多数为调试脚本或环境设置用途，通常可以忽略（除非你在做硬件调试或环境排错）。

## 核心文件一览（仅这些需要重点阅读）

- `data_coll/sync_data_collector.py`
  - 采集底层组件：配置读取、相机读帧、编码器读数、时间戳与缓存结构等基础能力。
  - 提供 `CameraReader` / `EncoderReader` / `SensorFrame` / `SyncFrame` 等核心类。

- `host_computer/hand_collector.py`
  - 单手采集核心逻辑：
    - 双目/单目相机启动与缓冲
    - 编码器读取与校准
    - 时间对齐与数据组帧
    - 立体校正、可视化、实时写盘/内存模式
  - **库文件**，不直接作为入口运行。

- `host_computer/single_hand_collector.py`
  - 单手采集 CLI 入口（包装 `HandCollector`）。
  - 支持 **可视化** / **录制** 两种模式。
  - 适合新同事作为最小可运行入口。

- `host_computer/dual_hand_collector.py`
  - 双手采集 CLI 入口。
  - 同时启动左右手 `HandCollector` 并进行对齐，支持可视化/录制。

- `4B/test_button.py`
  - 三线按键模块测试脚本（GPIO4）。
  - 交互式菜单，含按键状态、计数、响应时间等测试。

- `4B/test_gpio_led.py`
  - RGB LED + 按键联合测试脚本（LED: GPIO22/27/23，按钮: GPIO17）。
  - 交互式菜单，覆盖单色/混色/闪烁/按键联动。

> 说明：以上文件是项目的**核心采集/可视化/保存逻辑**与**必要的硬件测试入口**。除这些之外，其他文件大多是调试或环境配置用途，一般不需要深挖。

---

## 运行入口（Entry Points）

> 注意：本文档**不包含** `4B/gpio_data_collector.py` 的任何内容或说明。

### 1) 单手采集

```bash
python host_computer/single_hand_collector.py --config config.yaml --hand right
```

常用参数：
- `--hand left|right`：选择左手或右手（默认 `right`）
- `--visualize`：可视化模式（默认）
- `--record`：录制模式
- `--realtime-write`：实时写盘模式（默认 True，建议开启）

### 2) 双手采集
> **注：还没有按照类似“单手采集”的逻辑进行优化**

```bash
python host_computer/dual_hand_collector.py --config config.yaml --mode both --visualize
```

常用参数：
- `--mode left|right|both`：左右手或双手
- `--visualize`：可视化模式（默认）
- `--record`：录制模式

### 3) 按键测试

```bash
python 4B/test_button.py
```

- 默认按键引脚：GPIO4
- 三线按键模块接线：VCC(3.3V) / GND / OUT→GPIO4

### 4) LED + 按键联合测试

```bash
python 4B/test_gpio_led.py
```

- LED：GPIO22(红) / GPIO27(绿) / GPIO23(蓝)
- 按键：GPIO17

---

## 新人理解要点（极简）

- **核心采集流程**在 `hand_collector.py`，`single_hand_collector.py` / `dual_hand_collector.py` 只是入口壳。
- `sync_data_collector.py` 提供底层硬件读帧/读数能力，是所有采集逻辑的基础。
- 其它文件多数为调试或环境设置，通常可忽略。

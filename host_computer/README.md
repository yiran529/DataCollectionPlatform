# 双手数据收集系统

同时收集左右手的数据，每只手包括：
- 单目相机
- 双目相机
- 角度编码器

所有数据保证时间同步对齐。

## 硬件配置

### 相机连接

需要连接4个相机：
- **左手单目**: 设备ID 0 (在config.yaml中配置)
- **右手单目**: 设备ID 1 (在config.yaml中配置)
- **左手双目**: 设备ID 2 (在config.yaml中配置)
- **右手双目**: 设备ID 3 (在config.yaml中配置)

### 编码器连接

需要连接2个角度编码器：
- **左手编码器**: `/dev/ttyUSB0` (在config.yaml中配置)
- **右手编码器**: `/dev/ttyUSB1` (在config.yaml中配置)

## 配置文件

编辑 `config.yaml` 来配置相机和编码器参数：

```yaml
# 左手配置
left_hand:
  mono:
    device: 0          # 单目相机设备ID
    width: 1280       # 分辨率宽度
    height: 1024      # 分辨率高度
    fps: 30           # 帧率
  
  stereo:
    device: 2         # 双目相机设备ID
    width: 3840
    height: 1080
    fps: 30
  
  encoder:
    port: "/dev/ttyUSB0"      # 串口路径
    baudrate: 115200          # 波特率
    calibration:              # 编码器校准参数（直接配置）
      angle_zero: 120.443115234375  # 零点角度
      calibrated: true         # 是否已校准

# 右手配置
right_hand:
  mono:
    device: 1
    width: 1280
    height: 1024
    fps: 30
  
  stereo:
    device: 3
    width: 3840
    height: 1080
    fps: 30
  
  encoder:
    port: "/dev/ttyUSB1"
    baudrate: 115200
    calibration:              # 编码器校准参数（直接配置）
      angle_zero: 60.479736328125   # 零点角度
      calibrated: true         # 是否已校准

# 同步参数
sync:
  warmup_time: 1.0          # 预热时间（秒）
  calib_time: 0.5          # 校准时间（秒）
  max_time_diff_ms: 50.0   # 最大时间差（毫秒）

# 保存参数
save:
  output_dir: "./data"     # 输出目录
  jpeg_quality: 85         # JPEG质量 (1-100)
```

## 使用方法

### 单手数据收集（推荐）

使用 `single_hand_collector.py` 脚本，默认收集右手数据，使用更简单：

#### 右手可视化（默认）
```bash
cd /home/cjq/Documents/DataCollectionPlatform/host_computer
python3 single_hand_collector.py --visualize
```

#### 右手录制（默认）
```bash
python3 single_hand_collector.py --record
```

#### 左手可视化
```bash
python3 single_hand_collector.py --hand left --visualize
```

#### 左手录制
```bash
python3 single_hand_collector.py --hand left --record
```

**命令行参数**：
- `--hand, -H`: 选择左手或右手（`left` 或 `right`，默认：`right`）
- `--config, -c`: 配置文件路径（默认: `config.yaml`）
- `--visualize, -v`: 启用可视化模式
- `--record, -r`: 启用录制模式

### 双手数据收集

使用 `dual_hand_collector.py` 脚本进行双手数据收集：

#### 双手可视化
```bash
cd /home/cjq/Documents/DataCollectionPlatform/host_computer
python3 dual_hand_collector.py --config config.yaml --mode both --visualize
```

#### 左手可视化
```bash
python3 dual_hand_collector.py --config config.yaml --mode left --visualize
```

#### 右手可视化
```bash
python3 dual_hand_collector.py --config config.yaml --mode right --visualize
```

可视化窗口会显示：
- **双目图像**：左右视图（或上下视图）
- **单目图像**：单目相机视图
- **角度曲线**：实时角度变化曲线和当前角度值
- **FPS**：实时帧率

按 `q` 键退出可视化。

### 录制模式

#### 单手录制（推荐）
```bash
# 右手录制（默认）
python3 single_hand_collector.py --record

# 左手录制
python3 single_hand_collector.py --hand left --record
```

#### 双手录制
```bash
python3 dual_hand_collector.py --config config.yaml --mode both --record
```

#### 使用dual_hand_collector进行单手录制
```bash
python3 dual_hand_collector.py --config config.yaml --mode left --record
python3 dual_hand_collector.py --config config.yaml --mode right --record
```

程序会：
1. 初始化所有相机和编码器
2. 预热和校准
3. 等待用户按回车开始录制
4. 录制中等待用户按回车停止
5. 自动保存数据到HDF5文件

### 命令行参数

- `--config, -c`: 配置文件路径（默认: config.yaml）
- `--mode, -m`: 测试模式
  - `left`: 仅测试左手
  - `right`: 仅测试右手
  - `both`: 测试双手（默认）
- `--visualize, -v`: 启用可视化模式
- `--record, -r`: 启用录制模式

**注意**：如果不指定 `--visualize` 或 `--record`，默认使用可视化模式。

### 自定义配置文件

```bash
python3 dual_hand_collector.py --config my_config.yaml --mode left --visualize
```

## 数据格式

保存的数据文件为HDF5格式。

### 单手数据格式（single_hand_collector.py）

文件名格式：`YYYYMMDD_HHMMSS_{left|right}_hand_data.h5`

数据集：
- `stereo_jpeg`: 双目图像（JPEG压缩，已校正）
- `mono_jpeg`: 单目图像（JPEG压缩）
- `angles`: 角度值数组（float32）
- `timestamps`: 同步时间戳数组（float64）
- `stereo_timestamps`: 双目相机时间戳数组（float64）
- `mono_timestamps`: 单目相机时间戳数组（float64）
- `encoder_timestamps`: 编码器时间戳数组（float64）

元数据：
- `n_frames`: 总帧数
- `hand`: 手部标识（`left` 或 `right`）
- `stereo_shape`: 双目图像形状
- `mono_shape`: 单目图像形状
- `jpeg_quality`: JPEG质量
- `created_at`: 创建时间

### 双手数据格式（dual_hand_collector.py）

文件名格式：`YYYYMMDD_HHMMSS_dual_hand_data.h5`

数据集：

#### 左手数据
- `left_stereo_jpeg`: 左手双目图像（JPEG压缩）
- `left_mono_jpeg`: 左手单目图像（JPEG压缩）
- `left_angles`: 左手角度值数组
- `left_timestamps`: 左手时间戳数组

#### 右手数据
- `right_stereo_jpeg`: 右手双目图像（JPEG压缩）
- `right_mono_jpeg`: 右手单目图像（JPEG压缩）
- `right_angles`: 右手角度值数组
- `right_timestamps`: 右手时间戳数组

#### 同步信息
- `sync_timestamps`: 双手同步时间戳数组

### 元数据
- `n_frames`: 总帧数
- `left_stereo_shape`: 左手双目图像形状
- `left_mono_shape`: 左手单目图像形状
- `right_stereo_shape`: 右手双目图像形状
- `right_mono_shape`: 右手单目图像形状
- `jpeg_quality`: JPEG质量
- `created_at`: 创建时间

## 读取数据示例

### 读取单手数据

```python
import h5py
import cv2
import numpy as np

# 读取单手HDF5文件
with h5py.File('data/20241227_120000_right_hand_data.h5', 'r') as f:
    n_frames = f.attrs['n_frames']
    hand = f.attrs['hand']  # 'left' 或 'right'
    
    # 读取第一帧
    stereo_jpeg = f['stereo_jpeg'][0]
    mono_jpeg = f['mono_jpeg'][0]
    
    # 解码图像
    stereo = cv2.imdecode(np.frombuffer(stereo_jpeg, np.uint8), cv2.IMREAD_COLOR)
    mono = cv2.imdecode(np.frombuffer(mono_jpeg, np.uint8), cv2.IMREAD_COLOR)
    
    # 读取角度和时间戳
    angle = f['angles'][0]
    timestamp = f['timestamps'][0]
    
    print(f"帧数: {n_frames}")
    print(f"手部: {hand}")
    print(f"角度: {angle:.2f}°")
    print(f"时间戳: {timestamp}")
```

### 读取双手数据

```python
import h5py
import cv2
import numpy as np

# 读取双手HDF5文件
with h5py.File('data/20241227_120000_dual_hand_data.h5', 'r') as f:
    n_frames = f.attrs['n_frames']
    
    # 读取第一帧
    left_stereo_jpeg = f['left_stereo_jpeg'][0]
    left_mono_jpeg = f['left_mono_jpeg'][0]
    right_stereo_jpeg = f['right_stereo_jpeg'][0]
    right_mono_jpeg = f['right_mono_jpeg'][0]
    
    # 解码图像
    left_stereo = cv2.imdecode(np.frombuffer(left_stereo_jpeg, np.uint8), cv2.IMREAD_COLOR)
    left_mono = cv2.imdecode(np.frombuffer(left_mono_jpeg, np.uint8), cv2.IMREAD_COLOR)
    right_stereo = cv2.imdecode(np.frombuffer(right_stereo_jpeg, np.uint8), cv2.IMREAD_COLOR)
    right_mono = cv2.imdecode(np.frombuffer(right_mono_jpeg, np.uint8), cv2.IMREAD_COLOR)
    
    # 读取角度
    left_angle = f['left_angles'][0]
    right_angle = f['right_angles'][0]
    
    # 读取时间戳
    timestamp = f['sync_timestamps'][0]
    
    print(f"帧数: {n_frames}")
    print(f"左手角度: {left_angle:.2f}°")
    print(f"右手角度: {right_angle:.2f}°")
    print(f"时间戳: {timestamp}")
```

## 同步机制

程序使用时间戳对齐机制确保数据同步：

1. **单手指内同步**: 每只手的单目、双目、编码器数据基于时间戳对齐
2. **双手间同步**: 左右手数据基于时间戳对齐，最大时间差由 `max_time_diff_ms` 控制

对齐流程：
- 以左手双目相机的时间戳为基准
- 在右手数据中找到最接近时间戳的帧
- 如果时间差超过阈值，则丢弃该帧

## 故障排除

### 相机无法打开

1. 检查相机设备ID是否正确
2. 确认相机没有被其他程序占用
3. 检查USB连接

### 编码器无法连接

1. 检查串口路径是否正确（`/dev/ttyUSB0`, `/dev/ttyUSB1`）
2. 确认串口权限：`sudo chmod 666 /dev/ttyUSB*`
3. 检查波特率是否匹配

### 数据不同步

1. 增加 `warmup_time` 和 `calib_time`
2. 检查相机帧率是否稳定
3. 增加 `max_time_diff_ms` 阈值（但会降低同步精度）

## 依赖

- OpenCV (`cv2`)
- NumPy
- PyYAML
- h5py
- minimalmodbus (用于编码器)

安装依赖：
```bash
pip install opencv-python numpy pyyaml h5py minimalmodbus
```


# 双手数据采集完整指南

## 准备工作

### 1. 硬件连接检查

确保以下设备已正确连接：

#### 相机设备（4个）
- **左手单目相机**: 设备ID 8 (在config.yaml中配置)
- **左手双目相机**: 设备ID 10 (在config.yaml中配置)
- **右手单目相机**: 设备ID 6 (在config.yaml中配置)
- **右手双目相机**: 设备ID 4 (在config.yaml中配置)

#### 编码器设备（2个）
- **左手编码器**: `/dev/ttyUSB0` (在config.yaml中配置)
- **右手编码器**: `/dev/ttyUSB1` (在config.yaml中配置)

#### 检查设备连接
```bash
# 检查相机设备
ls /dev/video*

# 检查编码器串口
ls /dev/ttyUSB*
```

### 2. 标定准备（如果还未完成）

#### 双目相机标定

**左手双目标定**（如果已完成可跳过）:
```bash
cd /home/cjq/Documents/DataCollectionPlatform/host_computer

# 1. 采集标定图像
python3 stereo_calibration.py --capture --hand left --checkerboard 11x8 --square_size 25

# 2. 执行标定
python3 stereo_calibration.py --calibrate --hand left --checkerboard 11x8

# 3. 验证标定效果
python3 stereo_calibration.py --verify --hand left
```

**右手双目标定**（如果已完成可跳过）:
```bash
# 1. 采集标定图像
python3 stereo_calibration.py --capture --hand right --checkerboard 11x8 --square_size 25

# 2. 执行标定
python3 stereo_calibration.py --calibrate --hand right --checkerboard 11x8

# 3. 验证标定效果
python3 stereo_calibration.py --verify --hand right
```

标定结果会自动写入 `config.yaml`，无需手动配置。

### 3. 配置文件检查

确认 `config.yaml` 中的配置正确：

```yaml
left_hand:
  mono:
    device: 8        # 左手单目相机ID
    width: 1600
    height: 1200
    fps: 30
  stereo:
    device: 10       # 左手双目相机ID
    width: 3840
    height: 1080
    fps: 60
    calibration:     # 标定参数（标定后自动写入）
      calibrated: true
  encoder:
    port: "/dev/ttyUSB0"
    baudrate: 115200
    calibration:
      angle_zero: 120.443115234375
      calibrated: true

right_hand:
  mono:
    device: 6        # 右手单目相机ID
    width: 1280
    height: 1024
    fps: 30
  stereo:
    device: 4        # 右手双目相机ID
    width: 3840
    height: 1080
    fps: 30
    calibration:     # 标定参数（标定后自动写入）
      calibrated: true
  encoder:
    port: "/dev/ttyUSB1"
    baudrate: 115200
    calibration:
      angle_zero: 60.479736328125
      calibrated: true
```

## 数据采集流程

### 步骤1: 可视化测试（推荐先进行）

在正式采集前，先进行可视化测试，确保所有设备正常工作。

#### 双手可视化测试
```bash
cd /home/cjq/Documents/DataCollectionPlatform/host_computer
python3 dual_hand_collector.py --mode both --visualize
```

#### 单手可视化测试
```bash
# 左手测试
python3 dual_hand_collector.py --mode left --visualize

# 右手测试
python3 dual_hand_collector.py --mode right --visualize
```

**可视化窗口显示内容**:
- **双目图像**: 左右视图（已应用立体校正）
- **单目图像**: 单目相机视图
- **角度曲线**: 实时角度变化曲线和当前角度值
- **FPS**: 实时帧率

**检查要点**:
1. ✅ 所有相机图像正常显示
2. ✅ 双目图像左右对齐（如果已标定）
3. ✅ 编码器角度值正常变化
4. ✅ 帧率稳定（接近配置的FPS）

按 `q` 键退出可视化。

### 步骤2: 开始数据采集

#### 双手数据采集
```bash
cd /home/cjq/Documents/DataCollectionPlatform/host_computer
python3 dual_hand_collector.py --mode both --record
```

#### 单手数据采集
```bash
# 左手数据采集
python3 dual_hand_collector.py --mode left --record

# 右手数据采集
python3 dual_hand_collector.py --mode right --record
```

### 步骤3: 采集流程说明

程序运行后会按以下流程执行：

1. **初始化阶段**
   ```
   [left_hand] 初始化...
   [left_hand] ✓ 相机已启动
   [left_hand] ✓ 立体校正已加载: config.yaml
   [left_hand] ✓ 编码器连接成功
   [right_hand] 初始化...
   ...
   ```

2. **预热和校准阶段**
   ```
   [left_hand] 预热 1.0s...
   [left_hand] [1.0s] S:60.0fps M:30.0fps
   [left_hand] 校准 0.5s...
   [left_hand] ✓ 偏移量: S-M=2.5ms, S-E=1.2ms
   ```

3. **等待开始录制**
   ```
   所有设备已就绪，按回车开始录制...
   ```
   - 按 **回车键** 开始录制

4. **录制中**
   ```
   录制中... 按回车停止录制
   [录制] 已录制: 150 帧 (5.0 秒)
   ```
   - 程序会实时显示已录制的帧数和时间
   - 按 **回车键** 停止录制

5. **保存数据**
   ```
   [保存] 开始保存数据...
   [保存] 压缩图像...
   [保存] 写入HDF5...
   ✅ 保存完成: ./data/20241227_120000_dual_hand_data.h5
   ```

## 数据文件说明

### 文件位置
数据文件保存在 `./data/` 目录下，文件名格式：
```
YYYYMMDD_HHMMSS_dual_hand_data.h5
```

例如：`20241227_143022_dual_hand_data.h5`

### 数据格式（HDF5）

#### 数据集
- `left_stereo_jpeg`: 左手双目图像（JPEG压缩，已校正）
- `left_mono_jpeg`: 左手单目图像（JPEG压缩）
- `left_angles`: 左手角度值数组（float32）
- `left_timestamps`: 左手时间戳数组（float64）
- `right_stereo_jpeg`: 右手双目图像（JPEG压缩，已校正）
- `right_mono_jpeg`: 右手单目图像（JPEG压缩）
- `right_angles`: 右手角度值数组（float32）
- `right_timestamps`: 右手时间戳数组（float64）
- `sync_timestamps`: 双手同步时间戳数组（float64）

#### 元数据（属性）
- `n_frames`: 总帧数
- `left_stereo_shape`: 左手双目图像形状 `[height, width, channels]`
- `left_mono_shape`: 左手单目图像形状
- `right_stereo_shape`: 右手双目图像形状
- `right_mono_shape`: 右手单目图像形状
- `jpeg_quality`: JPEG质量
- `created_at`: 创建时间（ISO格式）

## 读取数据示例

```python
import h5py
import cv2
import numpy as np

# 读取HDF5文件
filepath = 'data/20241227_143022_dual_hand_data.h5'
with h5py.File(filepath, 'r') as f:
    # 读取元数据
    n_frames = f.attrs['n_frames']
    left_stereo_shape = f.attrs['left_stereo_shape']
    print(f"总帧数: {n_frames}")
    print(f"左手双目图像形状: {left_stereo_shape}")
    
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
    
    print(f"左手角度: {left_angle:.2f}°")
    print(f"右手角度: {right_angle:.2f}°")
    print(f"时间戳: {timestamp}")
```

## 常见问题

### 1. 相机无法打开

**症状**: 程序显示"无法打开相机设备"

**解决方法**:
- 检查 `config.yaml` 中的设备ID是否正确
- 确认相机没有被其他程序占用
- 检查USB连接
- 尝试重新插拔USB线

### 2. 编码器无法连接

**症状**: 程序显示"编码器连接失败"

**解决方法**:
```bash
# 检查串口是否存在
ls -l /dev/ttyUSB*

# 检查串口权限
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyUSB1

# 检查串口是否被占用
lsof /dev/ttyUSB0
```

### 3. 立体校正未加载

**症状**: 程序显示"未找到立体校正参数"

**解决方法**:
- 确认已完成双目相机标定
- 检查 `config.yaml` 中是否有 `calibration` 字段
- 如果标定参数在JSON文件中，确保路径正确

### 4. 数据不同步

**症状**: 左右手数据时间戳差异较大

**解决方法**:
- 增加 `config.yaml` 中的 `warmup_time` 和 `calib_time`
- 检查相机帧率是否稳定
- 适当增加 `max_time_diff_ms` 阈值（但会降低同步精度）

### 5. 录制时帧率下降

**症状**: 录制时帧率明显低于配置值

**解决方法**:
- 降低图像分辨率
- 降低JPEG质量（在 `config.yaml` 中设置 `jpeg_quality`）
- 检查系统资源使用情况

## 命令行参数

```bash
python3 dual_hand_collector.py [选项]
```

### 选项说明

- `--config, -c`: 配置文件路径（默认: `config.yaml`）
- `--mode, -m`: 采集模式
  - `left`: 仅采集左手数据
  - `right`: 仅采集右手数据
  - `both`: 采集双手数据（默认）
- `--visualize, -v`: 启用可视化模式（测试用）
- `--record, -r`: 启用录制模式（正式采集）

**注意**: 如果不指定 `--visualize` 或 `--record`，默认使用可视化模式。

## 完整采集示例

```bash
# 1. 进入目录
cd /home/cjq/Documents/DataCollectionPlatform/host_computer

# 2. 可视化测试（确保所有设备正常）
python3 dual_hand_collector.py --mode both --visualize
# 按 'q' 退出

# 3. 开始正式采集
python3 dual_hand_collector.py --mode both --record
# 按回车开始录制
# 按回车停止录制

# 4. 数据文件保存在 ./data/ 目录
ls -lh ./data/
```

## 数据验证

采集完成后，建议验证数据：

```python
import h5py
import numpy as np

filepath = 'data/20241227_143022_dual_hand_data.h5'
with h5py.File(filepath, 'r') as f:
    n_frames = f.attrs['n_frames']
    print(f"总帧数: {n_frames}")
    
    # 检查时间戳连续性
    timestamps = f['sync_timestamps'][:]
    time_diffs = np.diff(timestamps)
    print(f"平均时间间隔: {np.mean(time_diffs):.4f} 秒")
    print(f"时间间隔标准差: {np.std(time_diffs):.4f} 秒")
    
    # 检查角度范围
    left_angles = f['left_angles'][:]
    right_angles = f['right_angles'][:]
    print(f"左手角度范围: [{np.min(left_angles):.2f}, {np.max(left_angles):.2f}]")
    print(f"右手角度范围: [{np.min(right_angles):.2f}, {np.max(right_angles):.2f}]")
```

## 注意事项

1. **采集前务必进行可视化测试**，确保所有设备正常工作
2. **确保有足够的磁盘空间**，数据文件可能较大（取决于录制时长和分辨率）
3. **录制时保持设备稳定**，避免移动相机或编码器
4. **定期检查数据文件**，确保数据完整
5. **标定参数会自动应用**，采集的数据已经是校正后的图像




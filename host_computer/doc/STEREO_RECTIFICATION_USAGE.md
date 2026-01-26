# 立体校正功能使用说明

## 概述

数据收集程序现在支持自动应用立体校正。如果存在标定文件，程序会自动加载并应用校正，确保采集的数据是经过校正的（左右图像水平对齐）。

## 工作原理

1. **自动加载**: 程序启动时会自动查找并加载立体校正文件
2. **实时校正**: 在采集数据时，所有双目图像都会自动应用校正
3. **保存校正后的数据**: 保存到文件的数据是校正后的图像

## 配置

在 `config.yaml` 中，可以为每个手的双目相机配置校正文件路径：

```yaml
left_hand:
  stereo:
    device: 4
    width: 3840
    height: 1080
    fps: 30
    calibration_file: "stereo_calibration/stereo_calibration.json"  # 可选
```

如果不指定 `calibration_file`，程序会按以下顺序自动查找：
1. `host_computer/stereo_calibration/stereo_calibration.json`
2. `stereo_calibration/stereo_calibration.json`
3. 当前目录下的 `stereo_calibration/stereo_calibration.json`

## 使用流程

### 1. 执行标定（如果还没有）

```bash
cd /home/cjq/Documents/DataCollectionPlatform/host_computer

# 采集标定图像
python3 stereo_calibration.py --capture --checkerboard 11x8 --square_size 25

# 执行标定
python3 stereo_calibration.py --calibrate --checkerboard 11x8
```

标定结果会保存到 `stereo_calibration/stereo_calibration.json`

### 2. 运行数据收集

```bash
python3 dual_hand_collector.py --mode left --visualize
```

程序启动时会显示：
```
[left_hand] ✓ 立体校正已加载: stereo_calibration/stereo_calibration.json
[left_hand]   基线距离: 120.50 mm
[left_hand]   重投影误差: 0.234 像素
```

如果没有找到标定文件，会显示：
```
[left_hand] ⚠️ 未找到立体校正文件，将使用原始图像
[left_hand]   如需校正，请先运行标定: python stereo_calibration.py --calibrate
```

### 3. 验证校正效果

在可视化模式下，您可以看到校正后的图像：
- 左右图像应该水平对齐
- 同一水平线上的物体应该在左右图像中的同一行

## 技术细节

### 校正映射

程序使用 OpenCV 的 `cv2.remap()` 函数应用校正：
- 校正映射在初始化时计算一次（从标定文件加载）
- 运行时使用查找表（LUT）进行快速校正
- 校正过程对性能影响很小

### 数据流程

1. **原始图像**: 从相机获取原始双目图像（左右拼接）
2. **分割**: 将图像分割为左右两部分
3. **校正**: 对左右图像分别应用校正映射
4. **拼接**: 重新拼接校正后的图像
5. **保存**: 保存校正后的图像到文件

### 性能影响

- 校正过程使用硬件加速的查找表
- 对帧率影响很小（通常 < 5%）
- 内存占用：校正映射表约占用几MB内存

## 注意事项

1. **标定文件路径**: 确保标定文件路径正确，或使用相对路径
2. **图像尺寸**: 标定时的图像尺寸必须与采集时的图像尺寸一致
3. **相机设备**: 标定和采集必须使用同一个相机设备
4. **标定质量**: 标定质量直接影响校正效果，建议重投影误差 < 0.5 像素

## 故障排除

### 问题1: 校正文件未加载

**症状**: 程序显示"未找到立体校正文件"

**解决**:
- 检查标定文件是否存在
- 检查配置文件中的路径是否正确
- 确认标定文件格式正确（JSON格式）

### 问题2: 校正后图像变形

**症状**: 校正后的图像看起来变形或裁剪过多

**解决**:
- 重新执行标定，使用更多标定图像
- 检查标定时的图像尺寸是否与采集时一致
- 尝试调整标定参数（如 `alpha` 值）

### 问题3: 性能下降

**症状**: 采集帧率明显下降

**解决**:
- 检查图像尺寸是否过大
- 确认校正映射是否正确加载
- 检查系统资源使用情况

## 相关文件

- `stereo_calibration.py`: 标定工具
- `dual_hand_collector.py`: 数据收集程序（包含校正功能）
- `stereo_calibration/stereo_calibration.json`: 标定结果文件


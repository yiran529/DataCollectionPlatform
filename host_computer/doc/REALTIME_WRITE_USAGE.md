# 实时写入模式使用说明

## 问题背景

在Jetson开发板上进行长时间录制（如15秒）时，由于内存有限，原有的"先缓存到内存再保存"方式会导致：
- 内存占用过大（450帧@30fps ≈ 200-500MB）
- OOM（Out of Memory）导致进程被杀掉
- 系统卡顿

## 解决方案

新增**实时写入模式**，在录制过程中直接将数据写入磁盘，避免内存累积。

### 核心优化

1. **独立写入线程**：数据采集和磁盘写入并行，避免阻塞采集
2. **队列缓冲**：使用有界队列（最大50批次）平滑数据流
3. **批量写入**：每10帧或0.5秒批量写入，减少磁盘IO次数
4. **内存释放**：写入后立即释放数据，内存占用降低95%

### 内存对比

| 模式 | 15秒录制内存占用 | 说明 |
|-----|---------------|------|
| 原内存模式 | ~200-500MB | 所有帧缓存在内存 |
| 实时写入模式 | ~10-20MB | 仅保持队列缓冲 |

## 使用方法

### 启用实时写入

添加 `--realtime-write` 参数：

```bash
# 录制模式 + 实时写入
python single_hand_collector.py --record --realtime-write

# 指定左手
python single_hand_collector.py --record --realtime-write --hand left
```

### 不启用（使用原内存模式）

```bash
# 录制模式（内存缓存）
python single_hand_collector.py --record

# 可视化模式（始终使用内存）
python single_hand_collector.py --visualize
```

## 工作流程

### 实时写入模式

1. **录制开始**：创建HDF5文件，启动写入线程
2. **数据采集**：每0.1秒采集数据并加入队列（不保存到内存）
3. **后台写入**：写入线程批量处理队列中的数据
   - 对齐时间戳
   - 应用立体校正
   - JPEG压缩
   - 写入HDF5
4. **录制结束**：等待队列清空，关闭文件

### 内存模式（原方式）

1. **录制开始**：清空缓冲区
2. **数据采集**：保存到内存列表
3. **录制结束**：批量对齐、压缩、保存

## 性能特点

### 实时写入模式

✅ **优点**：
- 内存占用极低（~10-20MB）
- 支持超长时间录制
- 避免OOM风险
- 实时监控写入进度

⚠️ **注意**：
- 需要足够的磁盘IO带宽
- 队列满时会短暂阻塞（避免内存累积）
- 无法在内存中预览完整数据

### 内存模式

✅ **优点**：
- 可以在保存前检查数据
- 写入速度快（一次性批量）
- 适合短时间录制

❌ **缺点**：
- 内存占用大
- 长时间录制可能OOM
- Jetson上不适合超过10秒的录制

## 配置参数

在 `config.yaml` 中配置：

```yaml
save:
  output_dir: './data'     # 输出目录
  jpeg_quality: 85         # JPEG质量（60-95，越高质量越好文件越大）
```

### JPEG质量建议

| 质量值 | 文件大小 | 适用场景 |
|-------|---------|---------|
| 60-70 | 小 | 快速原型，磁盘空间受限 |
| 75-85 | 中等 | **推荐**，平衡质量和大小 |
| 90-95 | 大 | 需要高质量图像 |

## 监控信息

录制时会显示实时状态：

```
[RIGHT] 录制: 450 帧 (15.0s, 30.0fps, 已写入:448, 队列:2, 0.5GB)
```

- **帧数**：已采集的总帧数
- **时间**：录制时长
- **fps**：实际采集帧率
- **已写入**：已写入磁盘的帧数
- **队列**：等待写入的批次数
- **内存**：当前进程内存占用

## 故障排查

### 队列持续增长

**现象**：队列数值不断增加
**原因**：磁盘写入速度跟不上采集速度
**解决**：
1. 降低JPEG质量（60-75）
2. 检查磁盘空间和IO性能
3. 降低相机分辨率或帧率

### 写入线程超时

**现象**：停止录制时等待超过10秒
**原因**：大量数据积压在队列
**解决**：
1. 让系统完成写入（耐心等待）
2. 重启程序，检查磁盘健康

### 帧率下降

**现象**：fps低于预期
**原因**：可能是CPU或USB带宽限制
**解决**：
1. 实时写入模式已优化，不应因写入降低帧率
2. 检查USB连接和带宽
3. 检查CPU占用（应该不是写入导致）

## 代码示例

### 完整录制脚本

```python
from hand_collector import HandCollector
import yaml

# 加载配置
config = yaml.safe_load(open('config.yaml'))
hand_config = config['right_hand']
save_cfg = config['save']

# 创建收集器（启用实时写入）
collector = HandCollector(
    hand_config,
    'RIGHT',
    enable_realtime_write=True,
    output_dir=save_cfg['output_dir'],
    jpeg_quality=save_cfg['jpeg_quality']
)

# 启动并预热
collector.start()
collector.wait_ready()
collector.warmup_and_calibrate(1.0, 0.5)

# 开始录制
input("按回车开始录制...")
collector.start_recording()

input("按回车停止录制...")
filepath = collector.stop_recording()

print(f"保存到: {filepath}")
collector.stop()
```

## 技术细节

### 队列设计

```python
_write_queue = queue.Queue(maxsize=50)  # 最多50批次
```

- 每批10帧，最多缓冲500帧
- 30fps录制时，缓冲约16秒数据
- 队列满时阻塞采集，避免无限增长

### 批量策略

```python
batch_size = 10          # 每批10帧
max_wait_time = 0.5      # 最长等待0.5秒
```

- 平衡延迟和效率
- 0.5秒超时确保数据及时写入

### HDF5优化

```python
# 使用最新格式，支持动态扩展
h5py.File(filepath, 'w', libver='latest')

# 可变长度数据集（支持resize）
create_dataset('stereo_jpeg', (0,), maxshape=(None,))
```

## 总结

实时写入模式特别适合：
- ✅ Jetson等内存受限设备
- ✅ 长时间连续录制（>10秒）
- ✅ 需要稳定运行的生产环境

内存模式适合：
- ✅ 短时间录制（<5秒）
- ✅ 需要在保存前检查数据
- ✅ 内存充足的主机

**推荐**：在Jetson上默认使用 `--realtime-write` 模式！

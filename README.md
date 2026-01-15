# 数据采集平台

一个模块化、可扩展的数据采集平台，支持多种相机和编码器的视频流、位姿（Pose）和角度数据的实时采集与可视化。

## 功能特性

### 📹 视频流采集
- 支持多种相机类型（T265、ZED 2i、DECXIN 立体相机等）
- 实时视频帧预览
- 可配置采集分辨率和帧率

### 🎯 位姿追踪
- 6DoF位姿数据（位置 + 姿态）
- 3D实时可视化
- 轨迹记录与显示

### 📐 角度数据
- 支持多种编码器（SF11S03磁编码器等）
- 实时角度、速度、圈数显示
- 仪表盘可视化

### 💾 数据录制
- 视频、位姿、角度数据同步录制
- 双目图像序列保存（用于后处理 SLAM）
- JSON Lines格式存储
- 自动生成元数据

## 项目结构

```
DataCollectionPlatform/
├── backend/                    # 后端服务
│   ├── api/                    # API路由
│   │   └── routes.py           # FastAPI路由定义
│   ├── devices/                # 设备接口
│   │   ├── base.py             # 抽象基类
│   │   ├── camera_t265.py      # T265相机实现
│   │   ├── camera_zed2i.py     # ZED 2i相机实现
│   │   ├── camera_decxin.py    # DECXIN立体相机实现
│   │   ├── camera_mock.py      # 模拟相机（测试）
│   │   ├── encoder_sf11s03.py  # SF11S03编码器
│   │   └── encoder_mock.py     # 模拟编码器（测试）
│   ├── services/               # 业务服务
│   │   ├── data_collector.py   # 数据采集服务
│   │   └── recorder.py         # 录制服务
│   ├── config.py               # 配置管理
│   └── main.py                 # FastAPI应用入口
├── frontend/                   # 前端界面
│   └── dist/                   # 静态文件
├── post_process/               # 后处理脚本
│   └── evaluation_scripts/     # DROID-SLAM 评估脚本
│       └── test_decxin.py      # DECXIN 相机评估
├── tools/                      # 工具脚本
│   ├── decxin/                 # DECXIN 相机工具
│   │   ├── decxin_calibration.py   # 立体相机标定
│   │   └── decxin_camera_test.py   # 相机测试
│   ├── t265/                   # T265 相机工具
│   └── zed2i/                  # ZED 2i 相机工具
├── DROID-SLAM/                 # DROID-SLAM（需单独克隆，不包含在仓库中）
├── recordings/                 # 录制数据存储目录
├── requirements.txt            # Python依赖
├── run.py                      # 启动脚本
└── README.md                   # 说明文档
```

## 安装

### 1. 克隆主仓库

```bash
git clone https://github.com/your-repo/DataCollectionPlatform.git
cd DataCollectionPlatform
```

### 2. 安装主项目依赖

```bash
pip install -r requirements.txt
```

### 3. 安装相机驱动（按需）

**Intel RealSense T265:**
```bash
pip install pyrealsense2
```

**ZED 2i:**
```bash
# 安装 ZED SDK: https://www.stereolabs.com/developers/release
pip install pyzed
```

### 4. 安装 DROID-SLAM（可选，用于后处理 SLAM）

```bash
# 克隆 DROID-SLAM
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git

# 创建 conda 环境
cd DROID-SLAM
conda env create -f environment.yaml
conda activate droidenv

# 编译扩展
python setup.py install

# 下载模型权重
./tools/download_model.sh

cd ..
```

## 使用方法

### 启动数据采集平台

```bash
python run.py
```

访问 http://localhost:8000 打开 Web 界面。

### 立体相机标定

```bash
# 1. 采集标定图像（按空格键采集，需要采集 20 张左右）
python tools/decxin/decxin_calibration.py --capture --checkerboard 11x8 --square_size 25

# 2. 执行标定
python tools/decxin/decxin_calibration.py --calibrate

# 3. 验证标定结果
python tools/decxin/decxin_calibration.py --verify
```

标定结果保存在 `tools/decxin/calibration_images/stereo_calibration.json`。

### 录制数据

1. 启动数据采集平台：`python run.py`
2. 在 Web 界面配置相机类型为 `decxin`
3. 点击"开始采集"
4. 点击"开始录制"进行数据录制
5. 录制完成后，数据保存在 `recordings/session_xxx/` 目录

录制的数据结构：
```
recordings/session_xxx/
├── left/           # 左眼图像序列
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── right/          # 右眼图像序列
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── poses.jsonl     # 位姿数据
├── angles.jsonl    # 角度数据
├── video.avi       # 视频预览
└── metadata.json   # 元数据
```

### 运行 DROID-SLAM 后处理

使用 DROID-SLAM 处理录制的双目图像序列，生成相机轨迹：

```bash
# 激活 DROID-SLAM 环境
conda activate droidenv

# 运行评估脚本（可以从任意目录运行）
python post_process/evaluation_scripts/test_decxin.py \
    --datapath recordings/session_xxx \
    --stereo \
    --calib tools/decxin/calibration_images/stereo_calibration.json

# 或者指定输出文件
python post_process/evaluation_scripts/test_decxin.py \
    --datapath recordings/session_xxx \
    --stereo \
    --calib tools/decxin/calibration_images/stereo_calibration.json \
    --output my_trajectory.txt
```

**参数说明：**
- `--datapath`: 录制数据目录（包含 left/ 和 right/ 子目录）
- `--stereo`: 使用双目模式（推荐）
- `--calib`: 标定文件路径（可选，默认使用内置参数）
- `--output`: 输出轨迹文件路径（默认保存到数据目录中）
- `--disable_vis`: 禁用可视化窗口

输出的轨迹文件为 TUM 格式：`timestamp tx ty tz qx qy qz qw`

## 支持的设备

| 设备 | 类型 | 状态 |
|------|------|------|
| Intel RealSense T265 | 追踪相机 | ✅ 支持 |
| ZED 2i | 深度相机 | ✅ 支持 |
| DECXIN 立体相机 | 立体相机 | ✅ 支持 |
| SF11S03 | 磁编码器 | ✅ 支持 |

## API 接口

### WebSocket 数据流
- `ws://localhost:8000/ws`: 实时数据流（图像、位姿、角度）

### REST API
- `GET /api/status`: 系统状态
- `POST /api/configure/camera`: 配置相机
- `POST /api/configure/encoder`: 配置编码器
- `POST /api/collect/start`: 开始采集
- `POST /api/collect/stop`: 停止采集
- `POST /api/record/start`: 开始录制
- `POST /api/record/stop`: 停止录制

## License

MIT License

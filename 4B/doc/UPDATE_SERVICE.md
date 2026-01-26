# 更新服务以使用 Conda 环境

## 问题

服务启动时出现错误：
```
ModuleNotFoundError: No module named 'cv2'
```

这是因为服务使用的是系统 Python，而不是 conda 的 `box_tray` 环境。

## 解决方案

安装脚本已更新，会自动检测并使用 conda 的 `box_tray` 环境。

## 更新步骤

### 方法1：重新安装服务（推荐）

```bash
cd /home/ztlab/DataCollectionPlatform/4B

# 停止现有服务
sudo systemctl stop data_collector
sudo systemctl disable data_collector

# 重新安装服务（会自动使用 conda 环境）
sudo bash install_service.sh
```

### 方法2：手动更新服务文件

如果不想重新安装，可以手动更新服务文件：

```bash
# 编辑服务文件
sudo nano /etc/systemd/system/data_collector.service
```

将 `ExecStart` 行从：
```
ExecStart=/usr/bin/python3 ...
```

改为：
```
ExecStart=/home/ztlab/miniconda3/envs/box_tray/bin/python3 ...
```

并添加环境变量：
```
Environment="PATH=/home/ztlab/miniconda3/envs/box_tray/bin:/usr/local/bin:/usr/bin:/bin"
Environment="CONDA_DEFAULT_ENV=box_tray"
Environment="CONDA_PREFIX=/home/ztlab/miniconda3/envs/box_tray"
```

然后重新加载并重启：
```bash
sudo systemctl daemon-reload
sudo systemctl restart data_collector
```

## 验证

检查服务是否正常：

```bash
# 查看服务状态
sudo systemctl status data_collector

# 查看日志
tail -f /var/log/data_collector.log
```

应该不再出现 `ModuleNotFoundError` 错误。

## 说明

安装脚本现在会：
1. 自动检测 conda 环境 `box_tray`
2. 使用 conda 环境中的 Python 路径
3. 设置必要的环境变量（PATH, CONDA_DEFAULT_ENV, CONDA_PREFIX）

如果 conda 环境不存在，会回退到系统 Python3。






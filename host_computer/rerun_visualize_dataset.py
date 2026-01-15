#!/usr/bin/env python3
"""
使用 Rerun 可视化双手数据集 (HDF5)

依赖:
    pip install rerun-sdk h5py opencv-python numpy

用法示例:
    cd /home/cjq/Documents/DataCollectionPlatform/host_computer

    # 可视化最新的数据文件（自动在 ./data 下寻找最新的 .h5）
    python3 rerun_visualize_dataset.py

    # 指定文件
    python3 rerun_visualize_dataset.py --file ./data/20260106_145724_dual_hand_data.h5

    # 只看右手
    python3 rerun_visualize_dataset.py --file ./data/xxx.h5 --hand right

说明:
    - 左右手的双目图像会拆成 left/right 两张图像显示
    - 单目图像单独显示
    - 角度会以随时间变化的标量曲线形式显示
"""

import argparse
import glob
import os
import sys
import time
from typing import Optional

import h5py
import numpy as np
import cv2

try:
    import rerun as rr
except ImportError:
    rr = None


def find_latest_h5(data_dir: str) -> Optional[str]:
    pattern = os.path.join(data_dir, "*.h5")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def decode_jpeg_array(jpeg_bytes: np.ndarray) -> np.ndarray:
    """解码存储在 HDF5 vlen uint8 里的 JPEG 数据."""
    # jpeg_bytes 可能是 bytes 或 uint8 数组
    if isinstance(jpeg_bytes, bytes):
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    else:
        buf = np.asarray(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode JPEG image")
    # OpenCV 是 BGR，Rerun 默认用 RGB，这里转换一下
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def visualize_with_rerun(filepath: str, hand: str = "both", fps: float = 10.0):
    if rr is None:
        print("❌ 未安装 rerun-sdk，请先运行: pip install rerun-sdk")
        sys.exit(1)

    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        sys.exit(1)

    print(f"📂 打开数据文件: {filepath}")
    f = h5py.File(filepath, "r")

    # 检查是双手数据格式
    is_dual = all(k in f.keys() for k in ["left_stereo_jpeg", "right_stereo_jpeg"])

    if not is_dual:
        print("⚠️ 当前文件不是双手 HDF5 格式 (缺少 left/right_* 数据集)")
        f.close()
        sys.exit(1)

    n_frames = int(f.attrs.get("n_frames", len(f["left_stereo_jpeg"])))
    print(f"🧾 帧数: {n_frames}")

    # 读取角度和时间戳
    left_angles = f["left_angles"][:] if "left_angles" in f else None
    right_angles = f["right_angles"][:] if "right_angles" in f else None
    sync_ts = f["sync_timestamps"][:] if "sync_timestamps" in f else np.arange(n_frames, dtype=np.float64)

    # 初始化 Rerun
    rr.init("dual_hand_dataset", default_enabled=True)
    rr.spawn()  # 启动 Rerun Viewer

    # 设置时间序列（我们用帧号作为时间序列）
    timeline = "frame"

    def log_left(frame_idx: int):
        rr.set_time_sequence(timeline, frame_idx)  # 保持兼容性，虽然已弃用但还能用

        # 双目图像
        ls_jpeg = f["left_stereo_jpeg"][frame_idx]
        lm_jpeg = f["left_mono_jpeg"][frame_idx]

        stereo_img = decode_jpeg_array(ls_jpeg)
        mono_img = decode_jpeg_array(lm_jpeg)

        h, w = stereo_img.shape[:2]
        if w > h:
            left_img = stereo_img[:, : w // 2]
            right_img = stereo_img[:, w // 2 :]
        else:
            left_img = stereo_img[: h // 2, :]
            right_img = stereo_img[h // 2 :, :]

        # 使用 Rerun 新 API
        rr.log("left/stereo/left", rr.Image(left_img))
        rr.log("left/stereo/right", rr.Image(right_img))
        rr.log("left/mono", rr.Image(mono_img))

        if left_angles is not None:
            angle = float(left_angles[frame_idx])
            ts = float(sync_ts[frame_idx])
            # 以时间为横轴的标量曲线
            rr.set_time_seconds("time", ts)
            rr.log("left/angle", rr.Scalars(angle))

    def log_right(frame_idx: int):
        rr.set_time_sequence(timeline, frame_idx)  # 保持兼容性，虽然已弃用但还能用

        rs_jpeg = f["right_stereo_jpeg"][frame_idx]
        rm_jpeg = f["right_mono_jpeg"][frame_idx]

        stereo_img = decode_jpeg_array(rs_jpeg)
        mono_img = decode_jpeg_array(rm_jpeg)

        h, w = stereo_img.shape[:2]
        if w > h:
            left_img = stereo_img[:, : w // 2]
            right_img = stereo_img[:, w // 2 :]
        else:
            left_img = stereo_img[: h // 2, :]
            right_img = stereo_img[h // 2 :, :]

        # 使用 Rerun 新 API
        rr.log("right/stereo/left", rr.Image(left_img))
        rr.log("right/stereo/right", rr.Image(right_img))
        rr.log("right/mono", rr.Image(mono_img))

        if right_angles is not None:
            angle = float(right_angles[frame_idx])
            ts = float(sync_ts[frame_idx])
            rr.set_time_seconds("time", ts)
            rr.log("right/angle", rr.Scalars(angle))

    dt = 1.0 / fps if fps > 0 else 0.0

    print("🚀 开始发送数据到 Rerun Viewer (按 Ctrl+C 结束，仅影响本程序，数据文件不会修改)")
    try:
        for i in range(n_frames):
            if hand in ("left", "both"):
                log_left(i)
            if hand in ("right", "both"):
                log_right(i)

            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断播放")
    finally:
        f.close()
        print("✅ 结束，可在 Rerun Viewer 中继续浏览已发送的帧")


def main():
    parser = argparse.ArgumentParser(description="使用 Rerun 可视化双手 HDF5 数据集")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="",
        help="HDF5 数据文件路径（默认自动选择 ./data 下最新的 .h5）",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="数据目录（当未指定 --file 时，从此目录中查找最新文件）",
    )
    parser.add_argument(
        "--hand",
        type=str,
        choices=["left", "right", "both"],
        default="both",
        help="可视化哪只手的数据",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="播放帧率（仅影响发送到 Rerun 的速度，不改变原始时间戳）",
    )

    args = parser.parse_args()

    filepath = args.file
    if not filepath:
        filepath = find_latest_h5(args.data_dir)
        if not filepath:
            print(f"❌ 在目录 {args.data_dir} 下未找到任何 .h5 文件，请先录制数据")
            sys.exit(1)

    if not os.path.isabs(filepath):
        filepath = os.path.abspath(filepath)

    visualize_with_rerun(filepath, hand=args.hand, fps=args.fps)


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
将 stereo_frames_left 和 stereo_frames_right 文件夹中的图像转换为 H5 文件

使用方法:
    python convert_stereo_frames_to_h5.py [--left-dir stereo_frames_left] [--right-dir stereo_frames_right] [--output output.h5] [--quality 80]
"""

import os
import sys
import glob
import time
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional

try:
    import cv2
    import numpy as np
    import h5py
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    print(f"❌ 缺少依赖: {e}")
    sys.exit(1)

# TurboJPEG - 比OpenCV快2-5倍
try:
    from turbojpeg import TurboJPEG, TJPF_BGR
    TURBO_JPEG = TurboJPEG()
    HAS_TURBOJPEG = True
    print("✓ TurboJPEG 已启用（高速压缩）")
except ImportError:
    TURBO_JPEG = None
    HAS_TURBOJPEG = False
    print("ℹ 使用 OpenCV 进行 JPEG 压缩（建议安装 turbojpeg 以提高速度）")


def load_image_paths(left_dir: str, right_dir: str) -> Tuple[List[str], List[str]]:
    """
    加载左右图像文件路径，按文件名排序
    
    Returns:
        (left_paths, right_paths): 排序后的文件路径列表
    """
    # 获取所有 PNG 文件并排序
    left_pattern = os.path.join(left_dir, "left_*.png")
    right_pattern = os.path.join(right_dir, "right_*.png")
    
    left_paths = sorted(glob.glob(left_pattern))
    right_paths = sorted(glob.glob(right_pattern))
    
    # 验证数量是否匹配
    if len(left_paths) != len(right_paths):
        print(f"⚠️ 警告: 左图数量 ({len(left_paths)}) 与右图数量 ({len(right_paths)}) 不匹配")
        min_count = min(len(left_paths), len(right_paths))
        left_paths = left_paths[:min_count]
        right_paths = right_paths[:min_count]
        print(f"   使用前 {min_count} 对图像")
    
    return left_paths, right_paths


def save_stereo_frames_to_hdf5(left_dir: str, right_dir: str, output_path: str,
                               jpeg_quality: int = 80, n_workers: int = 4,
                               save_raw: bool = True) -> Optional[str]:
    """
    将左右图像文件夹转换为 H5 文件
    
    保存格式：
    - stereo_jpeg: JPEG压缩格式（节省空间）
    - stereo: 原始格式（兼容处理脚本，如果 save_raw=True）
    
    优化点：
    1. 并行读取图像
    2. 并行JPEG压缩（TurboJPEG优先）
    3. 超大批次写入（减少I/O次数）
    4. 预分配数据集
    """
    # 加载图像路径
    print(f"\n加载图像路径...")
    left_paths, right_paths = load_image_paths(left_dir, right_dir)
    
    if not left_paths or not right_paths:
        print("❌ 未找到图像文件")
        return None
    
    n_frames = len(left_paths)
    print(f"  找到 {n_frames} 对图像")
    
    # 生成输出文件名
    if not output_path:
        prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{prefix}_stereo_data.h5"
    
    encoder_name = "TurboJPEG" if HAS_TURBOJPEG else "OpenCV"
    print(f"\n保存: {output_path}")
    print(f"  帧数: {n_frames}, 编码器: {encoder_name}, 线程: {n_workers}")
    
    start_time = time.time()
    
    # 步骤1: 并行读取图像
    def load_image(idx: int):
        """加载单对图像"""
        left_img = cv2.imread(left_paths[idx])
        right_img = cv2.imread(right_paths[idx])
        
        if left_img is None:
            raise ValueError(f"无法读取图像: {left_paths[idx]}")
        if right_img is None:
            raise ValueError(f"无法读取图像: {right_paths[idx]}")
        
        return idx, left_img, right_img
    
    load_start = time.time()
    left_images = [None] * n_frames
    right_images = [None] * n_frames
    
    print("  读取图像...")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for idx, left_img, right_img in executor.map(load_image, range(n_frames)):
            left_images[idx] = left_img
            right_images[idx] = right_img
    
    load_time = time.time() - load_start
    fps_load = n_frames / load_time if load_time > 0 else 0
    print(f"  读取耗时: {load_time:.2f}s ({fps_load:.1f} fps)")
    
    # 获取图像尺寸
    left_shape = left_images[0].shape
    right_shape = right_images[0].shape
    print(f"  左图尺寸: {left_shape}, 右图尺寸: {right_shape}")
    
    # 合并左右图像为完整的 stereo 图像（左右并排）
    print("  合并左右图像为 stereo 图像...")
    stereo_images = []
    for i in range(n_frames):
        stereo_img = np.hstack([left_images[i], right_images[i]])  # 左右并排
        stereo_images.append(stereo_img)
    
    stereo_shape = stereo_images[0].shape
    print(f"  Stereo 图像尺寸: {stereo_shape} (宽度={stereo_shape[1]}, 左图={left_shape[1]}, 右图={right_shape[1]})")
    
    # 步骤2: 并行压缩所有帧到内存
    def encode_frame(idx: int):
        """压缩 stereo 帧到内存"""
        stereo_img = stereo_images[idx]
        
        if HAS_TURBOJPEG:
            stereo_jpg = TURBO_JPEG.encode(stereo_img, quality=jpeg_quality)
            return idx, stereo_jpg
        else:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            _, stereo_jpg = cv2.imencode('.jpg', stereo_img, encode_params)
            return idx, stereo_jpg.tobytes()
    
    compress_start = time.time()
    stereo_jpegs = [None] * n_frames
    
    print("  压缩图像...")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for idx, stereo_jpg in executor.map(encode_frame, range(n_frames)):
            stereo_jpegs[idx] = stereo_jpg
    
    compress_time = time.time() - compress_start
    fps_compress = n_frames / compress_time if compress_time > 0 else 0
    print(f"  压缩耗时: {compress_time:.2f}s ({fps_compress:.1f} fps)")
    
    # 步骤3: 优化HDF5写入
    write_start = time.time()
    
    # 计算最优chunk和batch大小
    chunk_size = min(500, max(100, n_frames // 4))
    batch_size = min(chunk_size * 2, max(500, n_frames // 2))
    
    print("  写入 HDF5...")
    with h5py.File(output_path, 'w', libver='latest', swmr=False) as f:
        # 元数据（与 gpio_data_collector.py 格式一致）
        f.attrs['n_frames'] = n_frames
        f.attrs['stereo_shape'] = stereo_shape
        f.attrs['jpeg_quality'] = jpeg_quality
        f.attrs['created_at'] = datetime.now().isoformat()
        f.attrs['source_left_dir'] = os.path.abspath(left_dir)
        f.attrs['source_right_dir'] = os.path.abspath(right_dir)
        # 保存原始左右图像尺寸信息
        f.attrs['left_shape'] = left_shape
        f.attrs['right_shape'] = right_shape
        
        # 创建可变长度数据集，使用优化的chunk
        dt = h5py.special_dtype(vlen=np.uint8)
        
        # 使用更大的chunk减少I/O（保存为 stereo_jpeg，与 gpio_data_collector.py 一致）
        stereo_ds = f.create_dataset(
            'stereo_jpeg',
            shape=(n_frames,),
            dtype=dt,
            chunks=(chunk_size,),
            compression=None,
            shuffle=False,  # 不shuffle（已经是压缩数据）
            fletcher32=False  # 不校验（加快写入）
        )
        
        # 超大批次写入（减少I/O次数）
        for i in range(0, n_frames, batch_size):
            end = min(i + batch_size, n_frames)
            # 批量转换（只在需要时转换）
            stereo_batch = [np.frombuffer(s, dtype=np.uint8) if isinstance(s, (bytes, bytearray)) else s 
                           for s in stereo_jpegs[i:end]]
            stereo_ds[i:end] = stereo_batch
        
        # 如果需要，同时保存原始格式的 stereo 键（兼容处理脚本）
        if save_raw:
            print("  同时保存原始格式 stereo 键（兼容处理脚本）...")
            stereo_raw_ds = f.create_dataset(
                'stereo',
                shape=(n_frames, *stereo_shape),
                dtype=np.uint8,
                chunks=(min(50, n_frames), *stereo_shape),  # 原始格式chunk较小
                compression='gzip',  # 使用压缩以减少文件大小
                compression_opts=1,  # 最小压缩级别（速度优先）
                shuffle=False,
                fletcher32=False
            )
            
            # 批量写入原始图像
            for i in range(0, n_frames, batch_size):
                end = min(i + batch_size, n_frames)
                stereo_raw_ds[i:end] = np.array(stereo_images[i:end], dtype=np.uint8)
    
    write_time = time.time() - write_start
    total_time = time.time() - start_time
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    speed = file_size / total_time if total_time > 0 else 0
    
    print(f"  写入耗时: {write_time:.2f}s")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  文件大小: {file_size:.1f}MB")
    print(f"  处理速度: {speed:.1f}MB/s")
    print(f"\n✅ 完成: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="将左右图像文件夹转换为 H5 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python convert_stereo_frames_to_h5.py
  python convert_stereo_frames_to_h5.py --left-dir ./stereo_frames_left --right-dir ./stereo_frames_right --output output.h5
  python convert_stereo_frames_to_h5.py --quality 95 --workers 8
  python convert_stereo_frames_to_h5.py --no-raw  # 只保存JPEG格式（节省空间）
        """
    )
    
    parser.add_argument("--left-dir", type=str, default="./stereo_frames_left",
                       help="左图像文件夹路径 (默认: ./stereo_frames_left)")
    parser.add_argument("--right-dir", type=str, default="./stereo_frames_right",
                       help="右图像文件夹路径 (默认: ./stereo_frames_right)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="输出 H5 文件路径 (默认: 自动生成)")
    parser.add_argument("--quality", "-q", type=int, default=80,
                       help="JPEG 质量 (1-100, 默认: 80)")
    parser.add_argument("--workers", "-w", type=int, default=4,
                       help="并行工作线程数 (默认: 4)")
    parser.add_argument("--save-raw", action="store_true", default=True,
                       help="同时保存原始格式 stereo 键（兼容处理脚本，默认: True）")
    parser.add_argument("--no-raw", action="store_false", dest="save_raw",
                       help="不保存原始格式，只保存 JPEG 格式（节省空间）")
    
    args = parser.parse_args()
    
    # 验证输入目录
    if not os.path.isdir(args.left_dir):
        print(f"❌ 左图像文件夹不存在: {args.left_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.right_dir):
        print(f"❌ 右图像文件夹不存在: {args.right_dir}")
        sys.exit(1)
    
    # 验证质量参数
    if not (1 <= args.quality <= 100):
        print(f"❌ JPEG 质量必须在 1-100 之间: {args.quality}")
        sys.exit(1)
    
    # 执行转换
    try:
        output_path = save_stereo_frames_to_hdf5(
            args.left_dir,
            args.right_dir,
            args.output,
            jpeg_quality=args.quality,
            n_workers=args.workers,
            save_raw=args.save_raw
        )
        
        if output_path:
            print(f"\n✅ 成功生成: {output_path}")
            sys.exit(0)
        else:
            print("\n❌ 转换失败")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


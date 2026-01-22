import h5py
import cv2
import numpy as np
import os
def analyze_h5_image_volumes(h5_filepath: str):
    """
    读取 HandCollector 存储的压缩图像，并计算解码后图像的文件大小。

    :param h5_filepath: HDF5 文件路径。
    """
    try:
        with h5py.File(h5_filepath, 'r') as h5_file:
            # 检查数据集是否存在
            if 'stereo_jpeg' not in h5_file or 'mono_jpeg' not in h5_file:
                print("HDF5 文件中缺少必要的数据集。")
                return

            # 读取压缩图像数据
            stereo_jpegs = h5_file['stereo_jpeg']
            mono_jpegs = h5_file['mono_jpeg']

            print(f"读取到 {len(stereo_jpegs)} 个 stereo_jpeg 和 {len(mono_jpegs)} 个 mono_jpeg。")

            # 计算文件大小
            stereo_sizes = []
            mono_sizes = []

            for i, jpeg_data in enumerate(stereo_jpegs):
                stereo_sizes.append(len(jpeg_data))

            for i, jpeg_data in enumerate(mono_jpegs):
                mono_sizes.append(len(jpeg_data))

            # 输出结果
            print("Stereo 图像文件大小统计:")
            print(f"  总数: {len(stereo_sizes)}")
            print(f"  平均大小: {np.mean(stereo_sizes):.2f} 字节")
            print(f"  总大小: {np.sum(stereo_sizes)} 字节")

            print("Mono 图像文件大小统计:")
            print(f"  总数: {len(mono_sizes)}")
            print(f"  平均大小: {np.mean(mono_sizes):.2f} 字节")
            print(f"  总大小: {np.sum(mono_sizes)} 字节")

            # 输出 HDF5 文件总大小
            file_size = os.path.getsize(h5_filepath)
            print(f"HDF5 文件总大小: {file_size} 字节")

    except Exception as e:
        print(f"读取 HDF5 文件时出错: {e}")

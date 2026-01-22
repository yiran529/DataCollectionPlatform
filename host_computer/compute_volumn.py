import h5py
import cv2
import numpy as np
def analyze_h5_image_volumes(h5_filepath: str):
    """
    读取 HandCollector 存储的压缩图像，并计算解码后图像的体积。

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

            # 解码并计算体积
            stereo_volumes = []
            mono_volumes = []

            for i, jpeg_data in enumerate(stereo_jpegs):
                img = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    volume = img.shape[0] * img.shape[1] * img.shape[2]
                    stereo_volumes.append(volume)

            for i, jpeg_data in enumerate(mono_jpegs):
                img = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    volume = img.shape[0] * img.shape[1] * img.shape[2]
                    mono_volumes.append(volume)

            # 输出结果
            print("Stereo 图像体积统计:")
            print(f"  总数: {len(stereo_volumes)}")
            print(f"  平均体积: {np.mean(stereo_volumes):.2f} 像素")

            print("Mono 图像体积统计:")
            print(f"  总数: {len(mono_volumes)}")
            print(f"  平均体积: {np.mean(mono_volumes):.2f} 像素")

    except Exception as e:
        print(f"读取 HDF5 文件时出错: {e}")

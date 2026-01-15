#!/usr/bin/env python3
"""
双目相机标定和校正工具

用于解决双目相机左右图像高度不一致的问题，生成立体校正参数

问题说明:
- 如果左右图像高度不一致，会影响双目SLAM的精度
- 通过立体标定和校正，可以将左右图像对齐到同一水平线上

使用流程:
1. 采集标定图像: python stereo_calibration.py --capture
2. 执行标定: python stereo_calibration.py --calibrate
3. 验证校正效果: python stereo_calibration.py --verify
"""

import cv2
import numpy as np
import os
import glob
import json
import argparse
import yaml
from datetime import datetime


class StereoCalibrator:
    """双目相机标定器"""
    
    def __init__(self, 
                 checkerboard_size=(11, 8),  # 棋盘格内角点数 (列, 行)
                 square_size=25.0,           # 方格边长 (mm)
                 calib_dir="stereo_calibration",
                 config_path=None,
                 hand="left"):  # "left" 或 "right"
        """
        Args:
            checkerboard_size: 棋盘格内角点数 (columns, rows)
            square_size: 方格边长 (mm)
            calib_dir: 标定图像保存目录
            config_path: 配置文件路径（用于获取相机参数）
            hand: 标定的手 ("left" 或 "right")
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.hand = hand.lower()
        self.calib_dir = calib_dir
        
        # 从配置文件读取相机参数（默认值）
        self.stereo_device = 4
        self.stereo_width = 3840
        self.stereo_height = 1080
        self.stereo_fps = 30
        self.mono_device = 0
        self.mono_width = 1280
        self.mono_height = 1024
        self.mono_fps = 30
        
        self.config_path = config_path
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # 根据hand参数选择对应的配置
                hand_key = f"{self.hand}_hand"
                hand_config = config.get(hand_key, {})
                
                # 读取双目相机配置
                stereo_cfg = hand_config.get('stereo', {})
                self.stereo_device = stereo_cfg.get('device', 4)
                self.stereo_width = stereo_cfg.get('width', 3840)
                self.stereo_height = stereo_cfg.get('height', 1080)
                self.stereo_fps = stereo_cfg.get('fps', 30)
                
                # 读取单目相机配置（用于信息显示）
                mono_cfg = hand_config.get('mono', {})
                self.mono_device = mono_cfg.get('device', 0)
                self.mono_width = mono_cfg.get('width', 1280)
                self.mono_height = mono_cfg.get('height', 1024)
                self.mono_fps = mono_cfg.get('fps', 30)
        
        # 单目图像尺寸（双目相机的一半）
        self.image_size = (self.stereo_width // 2, self.stereo_height)
        
        # 根据左右手创建不同的目录
        hand_calib_dir = os.path.join(calib_dir, self.hand)
        self.left_dir = os.path.join(hand_calib_dir, "left")
        self.right_dir = os.path.join(hand_calib_dir, "right")
        os.makedirs(self.left_dir, exist_ok=True)
        os.makedirs(self.right_dir, exist_ok=True)
        
        # 准备棋盘格的3D点
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size  # 转换为实际尺寸 (mm)
        
        # 标定结果
        self.calibration_result = None
    
    def capture_calibration_images(self, num_images=20):
        """采集标定图像"""
        print("=" * 60)
        print(f"{self.hand.upper()}手双目相机标定图像采集")
        print("=" * 60)
        print(f"双目相机设备ID: {self.stereo_device} (从config.yaml读取)")
        print(f"单目相机设备ID: {self.mono_device} (从config.yaml读取)")
        print(f"双目分辨率: {self.stereo_width}x{self.stereo_height}")
        print(f"单目分辨率: {self.mono_width}x{self.mono_height}")
        print(f"棋盘格尺寸: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} 内角点")
        print(f"方格边长: {self.square_size} mm")
        print(f"目标图像数: {num_images}")
        print(f"保存目录: {os.path.join(self.calib_dir, self.hand)}")
        print("=" * 60)
        print("\n操作说明:")
        print("  - 按 'c' 或 空格键: 捕获当前帧")
        print("  - 按 'q': 退出采集")
        print("  - 绿色角点: 检测成功")
        print("  - 红色文字: 检测失败")
        print("\n建议:")
        print("  - 将棋盘格放在不同位置、角度、距离拍摄")
        print("  - 确保棋盘格在左右图像中都清晰可见")
        print("  - 覆盖整个视野范围")
        print("=" * 60)
        
        # 打开相机
        print(f"\n正在打开双目相机设备 {self.stereo_device}...")
        cap = cv2.VideoCapture(self.stereo_device)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.stereo_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.stereo_height)
        cap.set(cv2.CAP_PROP_FPS, self.stereo_fps)
        
        if not cap.isOpened():
            print(f"❌ 无法打开相机设备 {self.stereo_device}!")
            return
        
        # 验证实际获取的分辨率
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✓ 相机已打开")
        print(f"  请求分辨率: {self.stereo_width}x{self.stereo_height}")
        print(f"  实际分辨率: {actual_width}x{actual_height}")
        print(f"  实际帧率: {actual_fps:.1f} fps")
        
        if actual_width < self.stereo_width * 0.9:
            print(f"⚠️ 警告: 实际宽度 ({actual_width}) 远小于请求宽度 ({self.stereo_width})")
            print(f"   这可能意味着打开的是单目相机而不是双目相机！")
            print(f"   请检查设备ID是否正确")
        
        captured_count = 0
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        while captured_count < num_images:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 检查图像尺寸
            if frame is None or frame.size == 0:
                continue
            
            frame_h, frame_w = frame.shape[:2]
            
            # 如果是第一次，显示图像尺寸信息
            if captured_count == 0:
                print(f"\n获取到的图像尺寸: {frame_w}x{frame_h}")
                if frame_w < self.stereo_width * 0.9:
                    print(f"⚠️ 警告: 图像宽度 ({frame_w}) 小于预期双目宽度 ({self.stereo_width})")
                    print(f"   这可能是单目相机！双目相机应该输出左右拼接的图像")
                    print(f"   请确认设备ID {self.stereo_device} 是否正确")
            
            # 分割左右图像
            mid = frame_w // 2
            left_img = frame[:, :mid]
            right_img = frame[:, mid:]
            
            # 显示左右图像尺寸（仅第一次）
            if captured_count == 0:
                print(f"左图像尺寸: {left_img.shape[1]}x{left_img.shape[0]}")
                print(f"右图像尺寸: {right_img.shape[1]}x{right_img.shape[0]}")
            
            # 转换为灰度图
            gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # 检测角点
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, self.checkerboard_size, None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, self.checkerboard_size, None)
            
            # 可视化
            display_l = left_img.copy()
            display_r = right_img.copy()
            
            if ret_l:
                corners_l_refined = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display_l, self.checkerboard_size, corners_l_refined, ret_l)
                cv2.putText(display_l, "LEFT: OK", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_l, "LEFT: NO CORNERS", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if ret_r:
                corners_r_refined = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display_r, self.checkerboard_size, corners_r_refined, ret_r)
                cv2.putText(display_r, "RIGHT: OK", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_r, "RIGHT: NO CORNERS", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 显示
            display = np.hstack([display_l, display_r])
            info_text = f"Captured: {captured_count}/{num_images} | Press 'c' to capture, 'q' to quit"
            cv2.putText(display, info_text, (10, display.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Stereo Calibration Capture", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif (key == ord('c') or key == ord(' ')) and ret_l and ret_r:
                # 保存图像
                left_path = os.path.join(self.left_dir, f"{captured_count:03d}.png")
                right_path = os.path.join(self.right_dir, f"{captured_count:03d}.png")
                cv2.imwrite(left_path, left_img)
                cv2.imwrite(right_path, right_img)
                captured_count += 1
                print(f"✓ 已保存图像对 {captured_count}/{num_images}")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n采集完成: {captured_count} 对图像")
        print(f"保存位置: {self.calib_dir}")
    
    def calibrate(self):
        """执行双目标定"""
        print("=" * 60)
        print(f"开始{self.hand.upper()}手双目相机标定")
        print("=" * 60)
        
        # 读取标定图像
        left_images = sorted(glob.glob(os.path.join(self.left_dir, "*.png")))
        right_images = sorted(glob.glob(os.path.join(self.right_dir, "*.png")))
        
        if len(left_images) == 0:
            print("❌ 未找到标定图像! 请先运行 --capture")
            return None
        
        if len(left_images) != len(right_images):
            print(f"⚠️ 左右图像数量不匹配: 左={len(left_images)}, 右={len(right_images)}")
            min_count = min(len(left_images), len(right_images))
            left_images = left_images[:min_count]
            right_images = right_images[:min_count]
        
        print(f"找到 {len(left_images)} 对标定图像")
        
        # 存储角点
        objpoints = []  # 3D 点
        imgpoints_l = []  # 左相机 2D 点
        imgpoints_r = []  # 右相机 2D 点
        
        # 角点检测标准
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        print("\n检测角点...")
        for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
            img_l = cv2.imread(left_path)
            img_r = cv2.imread(right_path)
            
            if img_l is None or img_r is None:
                print(f"  ✗ 图像 {i+1}: 无法读取")
                continue
            
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            
            # 检测角点
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, self.checkerboard_size, None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, self.checkerboard_size, None)
            
            if ret_l and ret_r:
                # 亚像素精度
                corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
                corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
                
                objpoints.append(self.objp)
                imgpoints_l.append(corners_l)
                imgpoints_r.append(corners_r)
                print(f"  ✓ 图像 {i+1}: 角点检测成功")
            else:
                print(f"  ✗ 图像 {i+1}: 角点检测失败 (左:{ret_l}, 右:{ret_r})")
        
        if len(objpoints) < 10:
            print(f"\n❌ 有效图像数量不足: {len(objpoints)} (建议至少10对)")
            return None
        
        print(f"\n有效图像: {len(objpoints)} 对")
        print("\n开始标定...")
        
        # 单目标定（分别标定左右相机）
        print("  1. 标定左相机...")
        ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
            objpoints, imgpoints_l, gray_l.shape[::-1], None, None
        )
        
        print("  2. 标定右相机...")
        ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
            objpoints, imgpoints_r, gray_r.shape[::-1], None, None
        )
        
        # 双目标定
        print("  3. 双目标定...")
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC  # 使用单目标定的内参
        
        ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_l, imgpoints_r,
            mtx_l, dist_l, mtx_r, dist_r,
            gray_l.shape[::-1],
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        
        if not ret_stereo:
            print("❌ 双目标定失败")
            return None
        
        print("  4. 计算立体校正参数...")
        # 计算立体校正参数
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            mtx_l, dist_l, mtx_r, dist_r,
            gray_l.shape[::-1], R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0  # 0=裁剪，1=保留所有像素
        )
        
        # 计算校正映射
        map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, gray_l.shape[::-1], cv2.CV_16SC2)
        map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, gray_r.shape[::-1], cv2.CV_16SC2)
        
        # 计算基线距离
        baseline = np.linalg.norm(T)
        
        # 保存标定结果
        result = {
            'left_camera_matrix': mtx_l.tolist(),
            'left_distortion': dist_l.tolist(),
            'right_camera_matrix': mtx_r.tolist(),
            'right_distortion': dist_r.tolist(),
            'rotation': R.tolist(),
            'translation': T.tolist(),
            'essential_matrix': E.tolist(),
            'fundamental_matrix': F.tolist(),
            'rectify_rotation_left': R1.tolist(),
            'rectify_rotation_right': R2.tolist(),
            'projection_left': P1.tolist(),
            'projection_right': P2.tolist(),
            'disparity_to_depth': Q.tolist(),
            'baseline_mm': float(baseline * self.square_size),  # 转换为mm
            'image_size': list(gray_l.shape[::-1]),
            'checkerboard_size': list(self.checkerboard_size),
            'square_size_mm': self.square_size,
            'calibration_date': datetime.now().isoformat(),
            'reprojection_error': float(ret_stereo)
        }
        
        # 保存校正映射（用于实时校正）
        # 根据左右手保存到不同的文件
        calibration_file = os.path.join(self.calib_dir, self.hand, f"stereo_calibration_{self.hand}.json")
        with open(calibration_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # 将标定结果写入config.yaml
        self._save_to_config_yaml(result)
        
        print(f"\n✅ {self.hand.upper()}手标定完成!")
        print(f"  重投影误差: {ret_stereo:.3f} 像素")
        print(f"  基线距离: {baseline * self.square_size:.2f} mm")
        print(f"  标定结果已保存: {calibration_file}")
        if hasattr(self, 'config_path') and self.config_path:
            print(f"  标定参数已写入: {self.config_path}")
        
        self.calibration_result = result
        return result
    
    def _save_to_config_yaml(self, calibration_result: dict):
        """将标定结果写入config.yaml"""
        if not hasattr(self, 'config_path') or not self.config_path or not os.path.exists(self.config_path):
            return
        
        try:
            # 读取现有配置
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # 准备标定参数（转换为列表格式，便于YAML保存）
            hand_key = f"{self.hand}_hand"
            if hand_key not in config:
                config[hand_key] = {}
            if 'stereo' not in config[hand_key]:
                config[hand_key]['stereo'] = {}
            
            # 将标定参数写入stereo配置
            stereo_cfg = config[hand_key]['stereo']
            stereo_cfg['calibration'] = {
                'left_camera_matrix': calibration_result['left_camera_matrix'],
                'left_distortion': calibration_result['left_distortion'],
                'right_camera_matrix': calibration_result['right_camera_matrix'],
                'right_distortion': calibration_result['right_distortion'],
                'rotation': calibration_result['rotation'],
                'translation': calibration_result['translation'],
                'rectify_rotation_left': calibration_result['rectify_rotation_left'],
                'rectify_rotation_right': calibration_result['rectify_rotation_right'],
                'projection_left': calibration_result['projection_left'],
                'projection_right': calibration_result['projection_right'],
                'disparity_to_depth': calibration_result['disparity_to_depth'],
                'baseline_mm': calibration_result['baseline_mm'],
                'image_size': calibration_result['image_size'],
                'reprojection_error': calibration_result['reprojection_error'],
                'calibration_date': calibration_result.get('calibration_date', ''),
                'calibrated': True
            }
            
            # 保存回文件
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
        except Exception as e:
            print(f"⚠️ 写入config.yaml失败: {e}")
            print("   标定结果已保存到JSON文件，但未写入config.yaml")
    
    def verify_calibration(self, device_id=None):
        """验证标定结果，显示校正后的图像"""
        if device_id is None:
            device_id = self.stereo_device
        
        # 加载标定结果（根据左右手）
        calibration_file = os.path.join(self.calib_dir, self.hand, f"stereo_calibration_{self.hand}.json")
        if not os.path.exists(calibration_file):
            print(f"❌ {self.hand.upper()}手标定文件不存在: {calibration_file}")
            print("   请先运行 --calibrate 进行标定")
            return
        
        with open(calibration_file, 'r') as f:
            calib = json.load(f)
        
        # 转换为numpy数组
        mtx_l = np.array(calib['left_camera_matrix'])
        dist_l = np.array(calib['left_distortion'])
        mtx_r = np.array(calib['right_camera_matrix'])
        dist_r = np.array(calib['right_distortion'])
        R1 = np.array(calib['rectify_rotation_left'])
        R2 = np.array(calib['rectify_rotation_right'])
        P1 = np.array(calib['projection_left'])
        P2 = np.array(calib['projection_right'])
        image_size = tuple(calib['image_size'])
        
        # 计算校正映射
        map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, image_size, cv2.CV_16SC2)
        map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, image_size, cv2.CV_16SC2)
        
        print("=" * 60)
        print(f"验证{self.hand.upper()}手标定结果")
        print("=" * 60)
        print(f"双目相机设备ID: {device_id} (从config.yaml读取)")
        print("校正后的图像应该:")
        print("  1. 左右图像水平对齐（同一行对应同一水平线）")
        print("  2. 畸变已校正")
        print("  3. 按 'q' 退出")
        print("=" * 60)
        
        # 打开相机
        cap = cv2.VideoCapture(device_id)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.stereo_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.stereo_height)
        cap.set(cv2.CAP_PROP_FPS, self.stereo_fps)
        
        if not cap.isOpened():
            print(f"❌ 无法打开相机设备 {device_id}!")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 分割左右图像
            mid = frame.shape[1] // 2
            left_raw = frame[:, :mid]
            right_raw = frame[:, mid:]
            
            # 应用校正
            left_rectified = cv2.remap(left_raw, map1_l, map2_l, cv2.INTER_LINEAR)
            right_rectified = cv2.remap(right_raw, map1_r, map2_r, cv2.INTER_LINEAR)
            
            # 绘制水平线用于验证对齐
            for y in range(0, left_rectified.shape[0], 50):
                cv2.line(left_rectified, (0, y), (left_rectified.shape[1], y), (0, 255, 0), 1)
                cv2.line(right_rectified, (0, y), (right_rectified.shape[1], y), (0, 255, 0), 1)
            
            # 显示
            display_raw = np.hstack([left_raw, right_raw])
            display_rectified = np.hstack([left_rectified, right_rectified])
            display = np.vstack([display_raw, display_rectified])
            
            cv2.putText(display, "RAW (Top) | RECTIFIED (Bottom) - Press 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Stereo Calibration Verification", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n验证完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="双目相机标定工具")
    parser.add_argument("--capture", action="store_true", help="采集标定图像")
    parser.add_argument("--calibrate", action="store_true", help="执行标定")
    parser.add_argument("--verify", action="store_true", help="验证标定结果")
    parser.add_argument("--hand", type=str, choices=["left", "right"], default="left", help="标定的手 (left/right)")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--device", "-d", type=int, help="相机设备ID（覆盖配置文件）")
    parser.add_argument("--num_images", "-n", type=int, default=20, help="采集图像数量")
    parser.add_argument("--checkerboard", type=str, default="11x8", help="棋盘格内角点数 (列x行)")
    parser.add_argument("--square_size", type=float, default=25.0, help="方格边长 (mm)")
    parser.add_argument("--output", "-o", type=str, default="stereo_calibration", help="输出目录")
    
    args = parser.parse_args()
    
    # 解析棋盘格尺寸
    cb_size = tuple(map(int, args.checkerboard.split('x')))
    
    # 获取配置文件路径
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    calibrator = StereoCalibrator(
        checkerboard_size=cb_size,
        square_size=args.square_size,
        calib_dir=args.output,
        config_path=config_path if os.path.exists(config_path) else None,
        hand=args.hand
    )
    
    # 如果指定了设备ID，覆盖配置
    if args.device is not None:
        calibrator.stereo_device = args.device
        print(f"⚠️ 使用命令行指定的双目相机设备ID: {args.device} (覆盖config.yaml中的值)")
        print(f"⚠️ 使用命令行指定的双目相机设备ID: {args.device} (覆盖config.yaml中的值)")
    
    if args.capture:
        calibrator.capture_calibration_images(args.num_images)
    elif args.calibrate:
        calibrator.calibrate()
    elif args.verify:
        device = args.device if args.device is not None else calibrator.stereo_device
        calibrator.verify_calibration(device)
    else:
        print("请指定操作: --capture, --calibrate, 或 --verify")
        print("\n使用流程:")
        print("  1. 采集标定图像（左手）:")
        print("     python stereo_calibration.py --capture --hand left --checkerboard 11x8 --square_size 25")
        print("  2. 执行标定（左手）:")
        print("     python stereo_calibration.py --calibrate --hand left --checkerboard 11x8")
        print("  3. 验证校正效果（左手）:")
        print("     python stereo_calibration.py --verify --hand left")
        print("\n  右手标定:")
        print("     python stereo_calibration.py --capture --hand right --checkerboard 11x8 --square_size 25")
        print("     python stereo_calibration.py --calibrate --hand right --checkerboard 11x8")
        print("     python stereo_calibration.py --verify --hand right")


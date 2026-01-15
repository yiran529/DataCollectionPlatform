#!/usr/bin/env python3
"""
DECXIN 立体相机标定脚本

使用棋盘格进行立体相机标定，获取 DROID-SLAM 所需的参数：
1. 内参矩阵 (K)
2. 畸变系数 (D)
3. 立体校正矩阵 (R1, R2, P1, P2)
4. 基线距离

使用方法:
1. 首先采集标定图像: python stereo_calibration.py --capture
2. 然后进行标定: python stereo_calibration.py --calibrate
"""

import cv2
import numpy as np
import os
import glob
import json
import argparse
from datetime import datetime


class StereoCalibrator:
    def __init__(self, 
                 checkerboard_size=(11, 8),  # 棋盘格内角点数 (列, 行)
                 square_size=25.0,           # 方格边长 (mm)
                 image_size=(1920, 1080),    # 单目图像尺寸
                 calib_dir="calibration_images"):
        """
        Args:
            checkerboard_size: 棋盘格内角点数 (columns, rows)
            square_size: 方格边长 (mm)
            image_size: 单目图像尺寸 (width, height)
            calib_dir: 标定图像保存目录
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.image_size = image_size
        self.calib_dir = calib_dir
        
        # 创建目录
        self.left_dir = os.path.join(calib_dir, "left")
        self.right_dir = os.path.join(calib_dir, "right")
        os.makedirs(self.left_dir, exist_ok=True)
        os.makedirs(self.right_dir, exist_ok=True)
        
        # 准备棋盘格的3D点
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size  # 转换为实际尺寸 (mm)
        
        # 标定结果
        self.calibration_result = None
    
    def capture_calibration_images(self, device_id=4, num_images=20):
        """
        采集标定图像
        
        Args:
            device_id: 相机设备ID
            num_images: 需要采集的图像数量
        """
        print("=" * 60)
        print("立体相机标定图像采集")
        print("=" * 60)
        print(f"棋盘格尺寸: {self.checkerboard_size[0]}x{self.checkerboard_size[1]} 内角点")
        print(f"方格边长: {self.square_size} mm")
        print(f"目标图像数: {num_images}")
        print(f"保存目录: {self.calib_dir}")
        print("=" * 60)
        print("\n操作说明:")
        print("  - 按 'c' 或 空格键: 捕获当前帧")
        print("  - 按 'q': 退出采集")
        print("  - 绿色角点: 检测成功")
        print("  - 红色文字: 检测失败")
        print("\n建议: 将棋盘格放在不同位置、角度、距离拍摄")
        print("=" * 60)
        
        # 打开相机
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ 无法打开相机!")
            return
        
        captured_count = 0
        
        while captured_count < num_images:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 分割左右图像
            mid = frame.shape[1] // 2
            left_img = frame[:, :mid]
            right_img = frame[:, mid:]
            
            # 转换为灰度图进行角点检测
            gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # 检测棋盘格角点
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret_l, corners_l = cv2.findChessboardCorners(gray_left, self.checkerboard_size, flags)
            ret_r, corners_r = cv2.findChessboardCorners(gray_right, self.checkerboard_size, flags)
            
            # 创建显示图像
            display_left = left_img.copy()
            display_right = right_img.copy()
            
            # 绘制检测结果
            if ret_l:
                cv2.drawChessboardCorners(display_left, self.checkerboard_size, corners_l, ret_l)
            else:
                cv2.putText(display_left, "Left: NOT DETECTED", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if ret_r:
                cv2.drawChessboardCorners(display_right, self.checkerboard_size, corners_r, ret_r)
            else:
                cv2.putText(display_right, "Right: NOT DETECTED", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 拼接显示
            display = np.hstack([display_left, display_right])
            display = cv2.resize(display, (1920, 540))
            
            # 显示状态
            status = f"Captured: {captured_count}/{num_images}"
            if ret_l and ret_r:
                status += " | READY (Press 'c' to capture)"
                color = (0, 255, 0)
            else:
                status += " | Move checkerboard"
                color = (0, 0, 255)
            
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Stereo Calibration", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c') or key == ord(' '):
                if ret_l and ret_r:
                    # 保存图像
                    filename = f"{captured_count:03d}.png"
                    cv2.imwrite(os.path.join(self.left_dir, filename), left_img)
                    cv2.imwrite(os.path.join(self.right_dir, filename), right_img)
                    captured_count += 1
                    print(f"✓ 已捕获第 {captured_count}/{num_images} 张图像")
                else:
                    print("⚠️ 角点未检测到，请调整棋盘格位置")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n采集完成! 共捕获 {captured_count} 张图像")
        print(f"保存位置: {self.calib_dir}")
    
    def calibrate(self):
        """
        执行立体标定
        """
        print("=" * 60)
        print("开始立体相机标定")
        print("=" * 60)
        
        # 读取标定图像
        left_images = sorted(glob.glob(os.path.join(self.left_dir, "*.png")))
        right_images = sorted(glob.glob(os.path.join(self.right_dir, "*.png")))
        
        if len(left_images) == 0:
            print("❌ 未找到标定图像! 请先运行 --capture")
            return None
        
        print(f"找到 {len(left_images)} 对标定图像")
        
        # 存储角点
        objpoints = []  # 3D 点
        imgpoints_l = []  # 左相机 2D 点
        imgpoints_r = []  # 右相机 2D 点
        
        # 角点检测标准
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
            img_l = cv2.imread(left_path)
            img_r = cv2.imread(right_path)
            
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
                print(f"  ✗ 图像 {i+1}: 角点检测失败")
        
        if len(objpoints) < 10:
            print(f"\n⚠️ 有效图像数量不足 ({len(objpoints)}/10)，建议重新采集更多图像")
        
        print(f"\n使用 {len(objpoints)} 对图像进行标定...")
        
        # ============================================================
        # 1. 单目标定 - 左相机
        # ============================================================
        print("\n[1/4] 标定左相机...")
        ret_l, K_l, D_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
            objpoints, imgpoints_l, self.image_size, None, None,
            flags=cv2.CALIB_FIX_K3
        )
        print(f"  左相机重投影误差: {ret_l:.4f} 像素")
        
        # ============================================================
        # 2. 单目标定 - 右相机
        # ============================================================
        print("\n[2/4] 标定右相机...")
        ret_r, K_r, D_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
            objpoints, imgpoints_r, self.image_size, None, None,
            flags=cv2.CALIB_FIX_K3
        )
        print(f"  右相机重投影误差: {ret_r:.4f} 像素")
        
        # ============================================================
        # 3. 立体标定
        # ============================================================
        print("\n[3/4] 立体标定...")
        flags = cv2.CALIB_FIX_INTRINSIC
        
        ret_stereo, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_l, imgpoints_r,
            K_l, D_l, K_r, D_r,
            self.image_size,
            criteria=criteria,
            flags=flags
        )
        print(f"  立体标定重投影误差: {ret_stereo:.4f} 像素")
        
        # ============================================================
        # 4. 立体校正
        # ============================================================
        print("\n[4/4] 计算立体校正参数...")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K_l, D_l, K_r, D_r,
            self.image_size, R, T,
            alpha=0,  # 0 = 只保留有效像素, 1 = 保留所有像素
            newImageSize=self.image_size
        )
        
        # 计算基线距离
        baseline = np.linalg.norm(T)
        
        # ============================================================
        # 保存结果
        # ============================================================
        self.calibration_result = {
            # 左相机参数
            "K_l": K_l.tolist(),
            "D_l": D_l.flatten().tolist(),
            "R1": R1.tolist(),
            "P1": P1.tolist(),
            
            # 右相机参数
            "K_r": K_r.tolist(),
            "D_r": D_r.flatten().tolist(),
            "R2": R2.tolist(),
            "P2": P2.tolist(),
            
            # 立体参数
            "R": R.tolist(),
            "T": T.flatten().tolist(),
            "E": E.tolist(),
            "F": F.tolist(),
            "Q": Q.tolist(),
            
            # 其他信息
            "baseline_mm": float(baseline),
            "image_size": list(self.image_size),
            "checkerboard_size": list(self.checkerboard_size),
            "square_size_mm": self.square_size,
            "reprojection_error": {
                "left": float(ret_l),
                "right": float(ret_r),
                "stereo": float(ret_stereo)
            },
            "roi1": list(roi1),
            "roi2": list(roi2)
        }
        
        # 保存为 JSON
        calib_file = os.path.join(self.calib_dir, "stereo_calibration.json")
        with open(calib_file, 'w') as f:
            json.dump(self.calibration_result, f, indent=2)
        
        # 生成 DROID-SLAM 格式的标定文件
        self._save_droid_calib()
        
        # 打印结果
        self._print_results()
        
        return self.calibration_result
    
    def _save_droid_calib(self):
        """保存 DROID-SLAM 格式的标定文件"""
        if self.calibration_result is None:
            return
        
        result = self.calibration_result
        
        # DROID-SLAM calib 格式: fx fy cx cy k1 k2 p1 p2
        # 使用校正后的内参 (从 P1 矩阵提取)
        P1 = np.array(result["P1"])
        fx = P1[0, 0]
        fy = P1[1, 1]
        cx = P1[0, 2]
        cy = P1[1, 2]
        
        # 校正后图像没有畸变
        k1, k2, p1, p2 = 0, 0, 0, 0
        
        # 保存 calib 文件
        calib_file = os.path.join(self.calib_dir, "decxin_stereo.txt")
        with open(calib_file, 'w') as f:
            f.write(f"{fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2}\n")
        
        print(f"\nDROID-SLAM 标定文件: {calib_file}")
        
        # 生成完整的 Python 代码片段
        code_file = os.path.join(self.calib_dir, "camera_params.py")
        with open(code_file, 'w') as f:
            f.write('"""\nDECXIN 立体相机参数\n自动生成于标定过程\n"""\n\n')
            f.write('import numpy as np\n\n')
            f.write(f'# 原始图像尺寸\n')
            f.write(f'ht0, wd0 = {result["image_size"][1]}, {result["image_size"][0]}\n\n')
            
            f.write('# 左相机内参\n')
            f.write(f'K_l = np.array({result["K_l"]}).reshape(3, 3)\n\n')
            
            f.write('# 左相机畸变系数\n')
            f.write(f'd_l = np.array({result["D_l"]})\n\n')
            
            f.write('# 左相机校正旋转矩阵\n')
            f.write(f'R_l = np.array({result["R1"]}).reshape(3, 3)\n\n')
            
            f.write('# 左相机投影矩阵\n')
            f.write(f'P_l = np.array({result["P1"]}).reshape(3, 4)\n\n')
            
            f.write('# 右相机内参\n')
            f.write(f'K_r = np.array({result["K_r"]}).reshape(3, 3)\n\n')
            
            f.write('# 右相机畸变系数\n')
            f.write(f'd_r = np.array({result["D_r"]})\n\n')
            
            f.write('# 右相机校正旋转矩阵\n')
            f.write(f'R_r = np.array({result["R2"]}).reshape(3, 3)\n\n')
            
            f.write('# 右相机投影矩阵\n')
            f.write(f'P_r = np.array({result["P2"]}).reshape(3, 4)\n\n')
            
            f.write('# 左到右的旋转矩阵\n')
            f.write(f'R = np.array({result["R"]}).reshape(3, 3)\n\n')
            
            f.write('# 左到右的平移向量 (mm)\n')
            f.write(f'T = np.array({result["T"]}).reshape(3, 1)\n\n')
            
            f.write(f'# 基线距离: {result["baseline_mm"]:.2f} mm\n')
            f.write(f'baseline = {result["baseline_mm"]}\n\n')
            
            f.write('# 校正后的内参向量 [fx, fy, cx, cy]\n')
            P1 = np.array(result["P1"])
            f.write(f'intrinsics_vec = [{P1[0,0]}, {P1[1,1]}, {P1[0,2]}, {P1[1,2]}]\n')
        
        print(f"Python 参数文件: {code_file}")
    
    def _print_results(self):
        """打印标定结果"""
        if self.calibration_result is None:
            return
        
        result = self.calibration_result
        
        print("\n" + "=" * 60)
        print("标定结果")
        print("=" * 60)
        
        K_l = np.array(result["K_l"])
        K_r = np.array(result["K_r"])
        P1 = np.array(result["P1"])
        
        print(f"\n左相机内参:")
        print(f"  fx = {K_l[0,0]:.2f}")
        print(f"  fy = {K_l[1,1]:.2f}")
        print(f"  cx = {K_l[0,2]:.2f}")
        print(f"  cy = {K_l[1,2]:.2f}")
        
        print(f"\n右相机内参:")
        print(f"  fx = {K_r[0,0]:.2f}")
        print(f"  fy = {K_r[1,1]:.2f}")
        print(f"  cx = {K_r[0,2]:.2f}")
        print(f"  cy = {K_r[1,2]:.2f}")
        
        print(f"\n立体参数:")
        print(f"  基线距离: {result['baseline_mm']:.2f} mm")
        print(f"  平移向量: {result['T']}")
        
        print(f"\n重投影误差:")
        print(f"  左相机: {result['reprojection_error']['left']:.4f} 像素")
        print(f"  右相机: {result['reprojection_error']['right']:.4f} 像素")
        print(f"  立体: {result['reprojection_error']['stereo']:.4f} 像素")
        
        print(f"\n校正后内参 (用于 DROID-SLAM):")
        print(f"  fx = {P1[0,0]:.2f}")
        print(f"  fy = {P1[1,1]:.2f}")
        print(f"  cx = {P1[0,2]:.2f}")
        print(f"  cy = {P1[1,2]:.2f}")
        
        print("\n" + "=" * 60)
        print(f"标定文件保存位置: {self.calib_dir}/")
        print("=" * 60)
    
    def verify_calibration(self, device_id=4):
        """
        验证标定结果 - 显示校正后的立体图像
        """
        # 加载标定结果
        calib_file = os.path.join(self.calib_dir, "stereo_calibration.json")
        if not os.path.exists(calib_file):
            print("❌ 未找到标定文件! 请先运行 --calibrate")
            return
        
        with open(calib_file, 'r') as f:
            result = json.load(f)
        
        K_l = np.array(result["K_l"])
        D_l = np.array(result["D_l"])
        R1 = np.array(result["R1"])
        P1 = np.array(result["P1"])
        
        K_r = np.array(result["K_r"])
        D_r = np.array(result["D_r"])
        R2 = np.array(result["R2"])
        P2 = np.array(result["P2"])
        
        # 创建校正映射
        map_l = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1[:3,:3], 
                                            tuple(result["image_size"]), cv2.CV_32F)
        map_r = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2[:3,:3], 
                                            tuple(result["image_size"]), cv2.CV_32F)
        
        print("=" * 60)
        print("标定验证 - 显示校正后的立体图像")
        print("=" * 60)
        print("绿色水平线应该穿过两幅图像中的同一物体")
        print("按 'q' 退出")
        print("=" * 60)
        
        # 打开相机
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 分割左右图像
            mid = frame.shape[1] // 2
            left_img = frame[:, :mid]
            right_img = frame[:, mid:]
            
            # 应用校正
            left_rect = cv2.remap(left_img, map_l[0], map_l[1], cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_img, map_r[0], map_r[1], cv2.INTER_LINEAR)
            
            # 拼接显示
            display = np.hstack([left_rect, right_rect])
            
            # 画水平线验证对齐
            for y in range(0, display.shape[0], 50):
                cv2.line(display, (0, y), (display.shape[1], y), (0, 255, 0), 1)
            
            display = cv2.resize(display, (1920, 540))
            cv2.imshow("Rectified Stereo", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DECXIN 立体相机标定")
    parser.add_argument("--capture", action="store_true", help="采集标定图像")
    parser.add_argument("--calibrate", action="store_true", help="执行标定")
    parser.add_argument("--verify", action="store_true", help="验证标定结果")
    parser.add_argument("--device", "-d", type=int, default=4, help="相机设备ID")
    parser.add_argument("--num_images", "-n", type=int, default=20, help="采集图像数量")
    parser.add_argument("--checkerboard", type=str, default="11x8", help="棋盘格内角点数 (列x行)")
    parser.add_argument("--square_size", type=float, default=25.0, help="方格边长 (mm)")
    parser.add_argument("--output", "-o", type=str, default="calibration_images", help="输出目录")
    
    args = parser.parse_args()
    
    # 解析棋盘格尺寸
    cb_size = tuple(map(int, args.checkerboard.split('x')))
    
    calibrator = StereoCalibrator(
        checkerboard_size=cb_size,
        square_size=args.square_size,
        calib_dir=args.output
    )
    
    if args.capture:
        calibrator.capture_calibration_images(args.device, args.num_images)
    elif args.calibrate:
        calibrator.calibrate()
    elif args.verify:
        calibrator.verify_calibration(args.device)
    else:
        print("请指定操作: --capture, --calibrate, 或 --verify")
        print("\n使用流程:")
        print("  1. python decxin_calibration.py --capture --checkerboard 11x8 --square_size 25")
        print("  2. python decxin_calibration.py --calibrate --checkerboard 11x8")
        print("  3. python decxin_calibration.py --verify")


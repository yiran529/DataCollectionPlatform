#!/usr/bin/env python3
"""
HDF5 数据测试工具
- 查看文件结构
- 预览图像
- 导出为MP4视频
"""

import h5py
import cv2
import numpy as np
import argparse
import os


def show_info(filepath: str):
    """显示HDF5文件信息"""
    print(f"\n{'='*60}")
    print(f"文件: {filepath}")
    print(f"大小: {os.path.getsize(filepath) / 1024 / 1024:.1f} MB")
    print(f"{'='*60}")
    
    with h5py.File(filepath, 'r') as f:
        # 属性
        print("\n📋 元数据:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        # 数据集
        print("\n📦 数据集:")
        for name in f.keys():
            ds = f[name]
            print(f"  {name}: shape={ds.shape}, dtype={ds.dtype}")
        
        # 快速模式检查（自动检测）
        fast_mode = 'stereo_jpeg' in f
        n_frames = f.attrs['n_frames']
        
        print(f"\n📊 统计:")
        print(f"  帧数: {n_frames}")
        print(f"  存储模式: {'JPEG压缩' if fast_mode else '原始像素'}")
        
        if 'angles' in f:
            angles = f['angles'][:]
            print(f"  角度范围: {angles.min():.1f} ~ {angles.max():.1f}")
        
        if 'timestamps' in f:
            ts = f['timestamps'][:]
            duration = ts[-1] - ts[0]
            fps = n_frames / duration if duration > 0 else 0
            print(f"  时长: {duration:.2f}s")
            print(f"  帧率: {fps:.1f} fps")


def decode_frame(f, idx: int, fast_mode: bool = None):
    """解码单帧（自动检测格式）"""
    # 自动检测：如果存在 stereo_jpeg 则使用 JPEG 模式
    if fast_mode is None:
        fast_mode = 'stereo_jpeg' in f
    
    if fast_mode:
        stereo = cv2.imdecode(np.array(f['stereo_jpeg'][idx]), cv2.IMREAD_COLOR)
        mono = cv2.imdecode(np.array(f['mono_jpeg'][idx]), cv2.IMREAD_COLOR)
    else:
        stereo = f['stereo'][idx]
        mono = f['mono'][idx]
    return stereo, mono


def preview(filepath: str, start_frame: int = 0):
    """预览图像"""
    print("\n预览模式 (按 q 退出, n 下一帧, p 上一帧, 空格播放)")
    
    with h5py.File(filepath, 'r') as f:
        fast_mode = 'stereo_jpeg' in f
        n_frames = f.attrs['n_frames']
        angles = f['angles'][:]
        timestamps = f['timestamps'][:]
        
        idx = start_frame
        playing = False
        
        while True:
            stereo, mono = decode_frame(f, idx, fast_mode)
            
            # 缩放显示
            h, w = stereo.shape[:2]
            stereo_small = cv2.resize(stereo, (w//2, h//2))
            mono_small = cv2.resize(mono, (mono.shape[1]//2, mono.shape[0]//2))
            
            # 信息叠加
            cv2.putText(stereo_small, f"Frame: {idx+1}/{n_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(stereo_small, f"Angle: {angles[idx]:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(stereo_small, f"Time: {timestamps[idx] - timestamps[0]:.2f}s", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("Stereo", stereo_small)
            cv2.imshow("Mono", mono_small)
            
            key = cv2.waitKey(30 if playing else 0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n') or key == 83:  # n or right arrow
                idx = min(idx + 1, n_frames - 1)
            elif key == ord('p') or key == 81:  # p or left arrow
                idx = max(idx - 1, 0)
            elif key == ord(' '):
                playing = not playing
            elif playing:
                idx = (idx + 1) % n_frames
        
        cv2.destroyAllWindows()


def export_video(filepath: str, output_path: str = None, fps: int = 30,
                 layout: str = "side"):
    """
    导出为MP4视频
    
    Args:
        filepath: HDF5文件路径
        output_path: 输出视频路径
        fps: 视频帧率
        layout: 布局方式
            - "side": 双目和单目左右并排
            - "stereo": 只导出双目
            - "mono": 只导出单目
            - "stack": 双目上下+单目右侧
    """
    if output_path is None:
        output_path = filepath.replace('.h5', '.mp4')
    
    print(f"\n导出视频: {output_path}")
    print(f"  帧率: {fps} fps")
    print(f"  布局: {layout}")
    
    with h5py.File(filepath, 'r') as f:
        fast_mode = 'stereo_jpeg' in f
        n_frames = f.attrs['n_frames']
        angles = f['angles'][:]
        timestamps = f['timestamps'][:]
        
        # 读取第一帧确定尺寸
        stereo, mono = decode_frame(f, 0, fast_mode)
        sh, sw = stereo.shape[:2]
        mh, mw = mono.shape[:2]
        
        # 计算输出尺寸
        if layout == "side":
            # 双目缩小 + 单目缩小，左右并排
            scale = 0.5
            out_h = int(max(sh, mh) * scale)
            out_w = int((sw + mw) * scale)
        elif layout == "stereo":
            out_h, out_w = sh // 2, sw // 2
        elif layout == "mono":
            out_h, out_w = mh, mw
        elif layout == "stack":
            # 双目上下叠 + 单目右侧
            out_h = sh
            mono_scaled_h = sh
            mono_scaled_w = int(mw * sh / mh)
            out_w = sw // 2 + mono_scaled_w
        else:
            print(f"未知布局: {layout}")
            return
        
        print(f"  输出尺寸: {out_w}x{out_h}")
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        
        if not writer.isOpened():
            print("❌ 无法创建视频文件")
            return
        
        for i in range(n_frames):
            stereo, mono = decode_frame(f, i, fast_mode)
            
            if layout == "side":
                # 缩放并并排
                stereo_s = cv2.resize(stereo, (int(sw * 0.5), int(sh * 0.5)))
                mono_s = cv2.resize(mono, (int(mw * 0.5), int(mh * 0.5)))
                
                # 高度对齐
                target_h = out_h
                if stereo_s.shape[0] != target_h:
                    stereo_s = cv2.resize(stereo_s, (stereo_s.shape[1], target_h))
                if mono_s.shape[0] != target_h:
                    mono_s = cv2.resize(mono_s, (mono_s.shape[1], target_h))
                
                frame = np.hstack([stereo_s, mono_s])
                
            elif layout == "stereo":
                frame = cv2.resize(stereo, (out_w, out_h))
                
            elif layout == "mono":
                frame = mono
                
            elif layout == "stack":
                # 分离左右眼
                left = stereo[:, :sw//2]
                right = stereo[:, sw//2:]
                
                # 缩放单目
                mono_s = cv2.resize(mono, (mono_scaled_w, mono_scaled_h))
                
                # 上下叠左右眼
                stereo_stack = np.vstack([
                    cv2.resize(left, (sw//2, sh//2)),
                    cv2.resize(right, (sw//2, sh//2))
                ])
                
                # 并排
                frame = np.hstack([stereo_stack, mono_s])
            
            # 确保尺寸正确
            if frame.shape[:2] != (out_h, out_w):
                frame = cv2.resize(frame, (out_w, out_h))
            
            # 添加信息
            cv2.putText(frame, f"Frame: {i+1}/{n_frames}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Angle: {angles[i]:.1f}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            writer.write(frame)
            
            if (i + 1) % 30 == 0:
                print(f"\r  进度: {i+1}/{n_frames}", end="", flush=True)
        
        writer.release()
        
        file_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"\r  完成: {n_frames}/{n_frames}")
        print(f"  文件大小: {file_size:.1f} MB")
        print(f"✅ 已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="HDF5数据测试工具")
    parser.add_argument("file", type=str, help="HDF5文件路径")
    parser.add_argument("--info", "-i", action="store_true", help="显示文件信息")
    parser.add_argument("--preview", "-p", action="store_true", help="预览图像")
    parser.add_argument("--video", "-v", action="store_true", help="导出为MP4视频")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出文件路径")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--layout", "-l", type=str, default="side",
                       choices=["side", "stereo", "mono", "stack"],
                       help="视频布局: side(并排), stereo(仅双目), mono(仅单目), stack(叠加)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"❌ 文件不存在: {args.file}")
        return
    
    # 默认显示信息
    show_info(args.file)
    
    if args.preview:
        preview(args.file)
    
    if args.video:
        export_video(args.file, args.output, args.fps, args.layout)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
双 GoPro 相机同步捕获

同时从两个 GoPro 捕获 1080p @ 60fps 图像，
支持偏移测量和补偿。
"""

import cv2
import numpy as np
import time
import os
import threading
import argparse
from queue import Queue
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, List

# 默认偏移量 (ms): cam1 比 cam2 慢的时间
DEFAULT_OFFSET_MS = 0.0


@dataclass
class FrameData:
    """帧数据"""
    frame: np.ndarray
    timestamp: float
    frame_idx: int
    camera_id: int


class CameraReader:
    """单个摄像头读取器（在独立线程中运行）"""
    
    def __init__(self, device_id: int, width: int, height: int, fps: int, name: str):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue: Queue = Queue(maxsize=30)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_count = 0
        self.start_time = 0.0
        
    def open(self) -> bool:
        """打开摄像头"""
        # 使用默认后端（不要用 V4L2，会限制帧率）
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            print(f"[{self.name}] 无法打开设备 {self.device_id}")
            return False
        
        # 必须先设置 MJPG 格式才能达到高帧率
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[{self.name}] 设备 {self.device_id}: {actual_w}x{actual_h} @ {actual_fps}fps")
        return True
    
    def start(self):
        """启动读取线程"""
        self.running = True
        self.frame_count = 0
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """停止读取"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
    
    def _read_loop(self):
        """读取循环"""
        # 预热
        for _ in range(5):
            self.cap.read()
        
        while self.running:
            ret, frame = self.cap.read()
            timestamp = time.time()
            
            if ret:
                self.frame_count += 1
                
                # 非阻塞放入队列，如果满了就丢弃最旧的
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                
                self.frame_queue.put(FrameData(
                    frame=frame,
                    timestamp=timestamp,
                    frame_idx=self.frame_count,
                    camera_id=self.device_id
                ))
    
    def get_frame(self, timeout: float = 0.1) -> Optional[FrameData]:
        """获取最新帧"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except:
            return None
    
    def get_latest_frame(self) -> Optional[FrameData]:
        """获取队列中最新的帧（清空队列，返回最后一个）"""
        latest = None
        while True:
            try:
                latest = self.frame_queue.get_nowait()
            except:
                break
        return latest
    
    def get_fps(self) -> float:
        """获取实际帧率"""
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0


def align_frames(cam1_data: FrameData, cam2_data: FrameData, 
                 output_height: int = 540, output_width: int = 1920) -> Tuple[np.ndarray, float]:
    """
    水平拼接两个相机图像到固定尺寸
    
    Returns:
        拼接后的图像(固定尺寸), 时间差(ms)
    """
    half_w = output_width // 2
    
    # 调整两个图像到相同尺寸
    img1 = cv2.resize(cam1_data.frame, (half_w, output_height))
    img2 = cv2.resize(cam2_data.frame, (half_w, output_height))
    
    # 水平拼接
    combined = np.hstack([img1, img2])
    
    # 计算时间差
    time_diff_ms = (cam1_data.timestamp - cam2_data.timestamp) * 1000
    
    return combined, time_diff_ms


def main():
    parser = argparse.ArgumentParser(description="双 GoPro 相机同步捕获")
    parser.add_argument("--cam1", type=int, default=0, help="GoPro 1 设备ID")
    parser.add_argument("--cam2", type=int, default=2, help="GoPro 2 设备ID")
    parser.add_argument("--width", type=int, default=1920, help="分辨率宽度")
    parser.add_argument("--height", type=int, default=1080, help="分辨率高度")
    parser.add_argument("--fps", type=int, default=60, help="帧率")
    
    parser.add_argument("--duration", "-t", type=int, default=10, help="录制时长(秒)")
    parser.add_argument("--output", "-o", type=str, default="gopro_sync", help="输出目录")
    parser.add_argument("--preview", "-p", action="store_true", help="显示实时预览")
    parser.add_argument("--offset", type=float, default=DEFAULT_OFFSET_MS,
                        help=f"CAM1 时间偏移量(ms)，默认{DEFAULT_OFFSET_MS}")
    parser.add_argument("--measure", "-m", action="store_true", help="测量偏移模式（不保存视频，只统计）")
    
    args = parser.parse_args()
    
    # 测量模式自动启用预览
    if args.measure:
        args.preview = True
    
    # 创建输出目录
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"capture_{timestamp_str}")
    if not args.measure:
        os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("双 GoPro 同步捕获" + (" [测量模式]" if args.measure else ""))
    print("=" * 60)
    print(f"CAM1: device={args.cam1}, {args.width}x{args.height}@{args.fps}fps")
    print(f"CAM2: device={args.cam2}, {args.width}x{args.height}@{args.fps}fps")
    print(f"时间偏移补偿: {args.offset:.1f} ms")
    print(f"录制时长: {args.duration} 秒")
    if not args.measure:
        print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 初始化摄像头
    cam1 = CameraReader(args.cam1, args.width, args.height, args.fps, "CAM1")
    cam2 = CameraReader(args.cam2, args.width, args.height, args.fps, "CAM2")
    
    if not cam1.open():
        return
    if not cam2.open():
        cam1.stop()
        return
    
    # 视频输出设置（固定尺寸，避免编码问题）
    output_height = 540
    output_width = 1920  # 固定宽度，每个相机 960
    output_fps = args.fps
    
    video_writer = None
    timestamp_file = None
    video_path = None
    timestamp_path = None
    
    if not args.measure:
        video_path = os.path.join(output_dir, "sync_video.mp4")
        # 使用 XVID 编码器，更兼容
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_path.replace('.mp4', '.avi'), fourcc, output_fps, (output_width, output_height))
        if not video_writer.isOpened():
            # 备选：使用 mp4v
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, output_fps, (output_width, output_height))
        
        timestamp_path = os.path.join(output_dir, "timestamps.txt")
        timestamp_file = open(timestamp_path, 'w')
        timestamp_file.write(f"# offset_ms={args.offset:.1f}\n")
        timestamp_file.write("# frame_idx, cam1_timestamp, cam2_timestamp, raw_diff_ms, corrected_diff_ms\n")
    
    print(f"\n预览: {output_width}x{output_height} @ {output_fps}fps")
    print(f"按 'q' 停止\n")
    
    # 启动摄像头
    cam1.start()
    cam2.start()
    
    # 等待摄像头稳定
    print("预热中...")
    time.sleep(2.0)
    
    frame_count = 0
    time_diffs = []
    raw_diffs = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < args.duration:
            # 获取帧
            cam1_data = cam1.get_frame(timeout=0.05)
            cam2_data = cam2.get_frame(timeout=0.05)
            
            if cam1_data is None or cam2_data is None:
                continue
            
            # 对齐并拼接（固定尺寸）
            combined, raw_diff_ms = align_frames(cam1_data, cam2_data, output_height, output_width)
            raw_diffs.append(raw_diff_ms)
            
            # 应用偏移补偿后的时间差
            corrected_diff_ms = raw_diff_ms - args.offset
            time_diffs.append(corrected_diff_ms)
            
            # 添加信息叠加
            info_text = f"Frame: {frame_count} | CAM1: {cam1.get_fps():.1f}fps | CAM2: {cam2.get_fps():.1f}fps"
            cv2.putText(combined, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            diff_color = (0, 255, 0) if abs(corrected_diff_ms) < 10 else (0, 165, 255) if abs(corrected_diff_ms) < 30 else (0, 0, 255)
            diff_text = f"Diff: {corrected_diff_ms:+.1f}ms (raw: {raw_diff_ms:+.1f}ms)"
            cv2.putText(combined, diff_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, diff_color, 2)
            
            # 标签
            cv2.putText(combined, "CAM1", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(combined, "CAM2", (output_width // 2 + 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 分割线
            cv2.line(combined, (output_width // 2, 0), (output_width // 2, output_height), (100, 100, 100), 2)
            
            # 写入视频
            if video_writer:
                video_writer.write(combined)
            
            # 记录时间戳
            if timestamp_file:
                timestamp_file.write(f"{frame_count}, {cam1_data.timestamp:.6f}, {cam2_data.timestamp:.6f}, {raw_diff_ms:.3f}, {corrected_diff_ms:.3f}\n")
            
            frame_count += 1
            
            # 预览
            if args.preview:
                cv2.imshow("GoPro Sync", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户中断")
                    break
            
            # 进度
            if frame_count % output_fps == 0:
                elapsed = time.time() - start_time
                avg_raw = np.mean(raw_diffs[-output_fps:])
                avg_corr = np.mean(time_diffs[-output_fps:])
                print(f"  [{elapsed:.0f}s] {frame_count}帧 | 原始偏移: {avg_raw:+.1f}ms | 校正后: {avg_corr:+.1f}ms")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        # 停止
        cam1.stop()
        cam2.stop()
        if video_writer:
            video_writer.release()
        if timestamp_file:
            timestamp_file.close()
        
        if args.preview:
            cv2.destroyAllWindows()
    
    # 统计
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("完成!" + (" [测量模式]" if args.measure else ""))
    print("=" * 60)
    print(f"总帧数: {frame_count}")
    print(f"录制时长: {total_time:.2f} 秒")
    print(f"实际帧率: {frame_count / total_time:.2f} fps")
    
    if raw_diffs:
        print(f"\n📊 原始时间差统计 (CAM1 - CAM2):")
        print(f"  平均值: {np.mean(raw_diffs):+.2f} ms")
        print(f"  中位数: {np.median(raw_diffs):+.2f} ms")
        print(f"  标准差: {np.std(raw_diffs):.2f} ms")
        print(f"  范围: {np.min(raw_diffs):+.2f} ~ {np.max(raw_diffs):+.2f} ms")
        
        if args.measure:
            # 测量模式：给出建议偏移量
            suggested_offset = np.median(raw_diffs)
            print(f"\n💡 建议偏移量: {suggested_offset:.1f} ms")
            print(f"   使用方式: python {os.path.basename(__file__)} --offset {suggested_offset:.1f}")
        else:
            # 正常模式：显示校正后结果
            print(f"\n📊 校正后时间差统计:")
            print(f"  应用偏移: {args.offset:.1f} ms")
            print(f"  平均误差: {np.mean(time_diffs):+.2f} ms")
            print(f"  标准差: {np.std(time_diffs):.2f} ms")
            
            avg_diff = abs(np.mean(time_diffs))
            if avg_diff < 10:
                print(f"\n✅ 同步质量: 优秀 (平均误差<10ms)")
            elif avg_diff < 20:
                print(f"\n✅ 同步质量: 良好 (平均误差<20ms)")
            elif avg_diff < 50:
                print(f"\n🔶 同步质量: 一般 (平均误差<50ms)")
            else:
                print(f"\n❌ 同步质量: 较差")
    
    if not args.measure and video_path:
        actual_video = video_path.replace('.mp4', '.avi') if os.path.exists(video_path.replace('.mp4', '.avi')) else video_path
        print(f"\n输出文件:")
        print(f"  视频: {actual_video}")
        print(f"  时间戳: {timestamp_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()


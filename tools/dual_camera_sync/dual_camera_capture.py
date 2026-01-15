#!/usr/bin/env python3
"""
双摄像头同步捕获

同时从两个摄像头捕获图像：
- 摄像头1: DECXIN 立体相机 (3840x1080 @ 60fps)
- 摄像头2: DECXIN 单目相机 (1280x1024 @ 60fps)

支持固定偏移补偿，将它们拼接并保存为视频。
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

# 默认偏移量 (ms): 立体相机比单目相机慢的时间
# 通过 analyze_timestamps.py 分析得出
DEFAULT_STEREO_MONO_OFFSET_MS = 35.0


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


def align_frames(stereo_data: FrameData, mono_data: FrameData, 
                 output_height: int = 720) -> Tuple[np.ndarray, float]:
    """
    对齐并拼接双目和单目图像
    
    布局:
    +------------------+--------+
    |    左图 (L)      |        |
    +------------------+  单目  |
    |    右图 (R)      |        |
    +------------------+--------+
    
    Returns:
        拼接后的图像, 时间差(ms)
    """
    stereo_frame = stereo_data.frame
    mono_frame = mono_data.frame
    
    # 分割立体图像
    mid = stereo_frame.shape[1] // 2
    left_img = stereo_frame[:, :mid]
    right_img = stereo_frame[:, mid:]
    
    # 调整尺寸
    # 左右图各 960x540，单目 480x720
    left_resized = cv2.resize(left_img, (960, output_height // 2))
    right_resized = cv2.resize(right_img, (960, output_height // 2))
    
    # 单目图像调整为右侧高度
    mono_h = output_height
    mono_w = int(mono_frame.shape[1] * mono_h / mono_frame.shape[0])
    mono_resized = cv2.resize(mono_frame, (mono_w, mono_h))
    
    # 垂直拼接左右图
    stereo_combined = np.vstack([left_resized, right_resized])
    
    # 水平拼接
    canvas_w = stereo_combined.shape[1] + mono_resized.shape[1]
    canvas = np.zeros((output_height, canvas_w, 3), dtype=np.uint8)
    
    canvas[:, :stereo_combined.shape[1]] = stereo_combined
    canvas[:, stereo_combined.shape[1]:] = mono_resized
    
    # 计算时间差
    time_diff_ms = (stereo_data.timestamp - mono_data.timestamp) * 1000
    
    return canvas, time_diff_ms


def main():
    parser = argparse.ArgumentParser(description="双摄像头同步捕获")
    parser.add_argument("--stereo-device", type=int, default=6, help="立体相机设备ID")
    parser.add_argument("--stereo-width", type=int, default=3840, help="立体相机宽度")
    parser.add_argument("--stereo-height", type=int, default=1080, help="立体相机高度")
    parser.add_argument("--stereo-fps", type=int, default=60, help="立体相机帧率")
    
    parser.add_argument("--mono-device", type=int, default=4, help="单目相机设备ID")
    parser.add_argument("--mono-width", type=int, default=1280, help="单目相机宽度")
    parser.add_argument("--mono-height", type=int, default=1024, help="单目相机高度")
    parser.add_argument("--mono-fps", type=int, default=60, help="单目相机帧率")
    
    parser.add_argument("--duration", "-t", type=int, default=10, help="录制时长(秒)")
    parser.add_argument("--output", "-o", type=str, default="sync_test", help="输出目录")
    parser.add_argument("--preview", "-p", action="store_true", help="显示实时预览")
    parser.add_argument("--offset", type=float, default=DEFAULT_STEREO_MONO_OFFSET_MS,
                        help=f"立体相机时间偏移量(ms)，默认{DEFAULT_STEREO_MONO_OFFSET_MS}")
    parser.add_argument("--no-offset", action="store_true", help="不应用偏移补偿（用于测量原始偏移）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"capture_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 偏移设置
    offset_ms = 0.0 if args.no_offset else args.offset
    
    print("=" * 60)
    print("双摄像头同步捕获")
    print("=" * 60)
    print(f"立体相机: device={args.stereo_device}, {args.stereo_width}x{args.stereo_height}@{args.stereo_fps}fps")
    print(f"单目相机: device={args.mono_device}, {args.mono_width}x{args.mono_height}@{args.mono_fps}fps")
    print(f"时间偏移补偿: {offset_ms:.1f} ms" + (" (已禁用)" if args.no_offset else ""))
    print(f"录制时长: {args.duration} 秒")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 初始化摄像头
    stereo_cam = CameraReader(args.stereo_device, args.stereo_width, args.stereo_height, 
                               args.stereo_fps, "STEREO")
    mono_cam = CameraReader(args.mono_device, args.mono_width, args.mono_height,
                            args.mono_fps, "MONO")
    
    if not stereo_cam.open():
        return
    if not mono_cam.open():
        stereo_cam.stop()
        return
    
    # 视频输出设置
    output_height = 720
    output_width = 960 + int(args.mono_width * output_height / args.mono_height)
    output_fps = min(args.stereo_fps, args.mono_fps)
    
    video_path = os.path.join(output_dir, "sync_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, output_fps, (output_width, output_height))
    
    # 时间戳文件
    timestamp_path = os.path.join(output_dir, "timestamps.txt")
    timestamp_file = open(timestamp_path, 'w')
    timestamp_file.write(f"# offset_ms={offset_ms:.1f}\n")
    timestamp_file.write("# frame_idx, stereo_timestamp, mono_timestamp, raw_diff_ms, corrected_diff_ms\n")
    
    print(f"\n视频输出: {output_width}x{output_height} @ {output_fps}fps")
    print(f"按 Ctrl+C 或 'q' 停止\n")
    
    # 启动摄像头
    stereo_cam.start()
    mono_cam.start()
    
    # 等待摄像头稳定
    time.sleep(0.5)
    
    frame_count = 0
    time_diffs = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < args.duration:
            # 获取帧（每个队列获取一帧，保持同步）
            stereo_data = stereo_cam.get_frame(timeout=0.05)
            mono_data = mono_cam.get_frame(timeout=0.05)
            
            if stereo_data is None or mono_data is None:
                continue
            
            # 对齐并拼接
            combined, raw_diff_ms = align_frames(stereo_data, mono_data, output_height)
            
            # 应用偏移补偿后的时间差
            corrected_diff_ms = raw_diff_ms - offset_ms
            time_diff_ms = corrected_diff_ms  # 用于显示和统计
            time_diffs.append(time_diff_ms)
            
            # 添加信息叠加
            info_text = f"Frame: {frame_count} | Stereo FPS: {stereo_cam.get_fps():.1f} | Mono FPS: {mono_cam.get_fps():.1f}"
            cv2.putText(combined, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            diff_color = (0, 255, 0) if abs(time_diff_ms) < 20 else (0, 165, 255) if abs(time_diff_ms) < 50 else (0, 0, 255)
            diff_text = f"Time Diff: {time_diff_ms:+.2f} ms"
            cv2.putText(combined, diff_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, diff_color, 2)
            
            # 标签
            cv2.putText(combined, "LEFT", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(combined, "RIGHT", (10, output_height // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(combined, "MONO", (970, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 分割线
            cv2.line(combined, (960, 0), (960, output_height), (100, 100, 100), 2)
            cv2.line(combined, (0, output_height // 2), (960, output_height // 2), (100, 100, 100), 1)
            
            # 写入视频
            video_writer.write(combined)
            
            # 记录时间戳
            timestamp_file.write(f"{frame_count}, {stereo_data.timestamp:.6f}, {mono_data.timestamp:.6f}, {raw_diff_ms:.3f}, {corrected_diff_ms:.3f}\n")
            
            frame_count += 1
            
            # 预览
            if args.preview:
                cv2.imshow("Dual Camera Sync", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n用户中断")
                    break
            
            # 进度
            if frame_count % output_fps == 0:
                elapsed = time.time() - start_time
                avg_diff = np.mean(time_diffs[-output_fps:])
                print(f"  已录制 {frame_count} 帧 ({elapsed:.1f}s), 平均时间差: {avg_diff:+.2f} ms")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    finally:
        # 停止
        stereo_cam.stop()
        mono_cam.stop()
        video_writer.release()
        timestamp_file.close()
        
        if args.preview:
            cv2.destroyAllWindows()
    
    # 统计
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("录制完成!")
    print("=" * 60)
    print(f"总帧数: {frame_count}")
    print(f"录制时长: {total_time:.2f} 秒")
    print(f"实际帧率: {frame_count / total_time:.2f} fps")
    
    if time_diffs:
        print(f"\n时间差统计 (补偿后):")
        print(f"  应用偏移: {offset_ms:.1f} ms")
        print(f"  平均误差: {np.mean(time_diffs):+.2f} ms")
        print(f"  标准差: {np.std(time_diffs):.2f} ms")
        print(f"  范围: {np.min(time_diffs):+.2f} ~ {np.max(time_diffs):+.2f} ms")
        
        # 判断同步质量
        avg_diff = abs(np.mean(time_diffs))
        std_diff = np.std(time_diffs)
        
        if avg_diff < 20 and std_diff < 30:
            print(f"\n✅ 同步质量: 良好 (平均误差<20ms)")
        elif avg_diff < 50 and std_diff < 50:
            print(f"\n🔶 同步质量: 一般 (平均误差<50ms)")
        else:
            print(f"\n❌ 同步质量: 较差 (建议重新校准偏移或使用硬件同步)")
    
    print(f"\n输出文件:")
    print(f"  视频: {video_path}")
    print(f"  时间戳: {timestamp_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()


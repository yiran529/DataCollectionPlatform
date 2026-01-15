#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编码器延迟测试工具（自动检测版）

自动测量编码器的采样延迟和响应时间

原理：
1. 持续高速采样编码器角度
2. 自动检测角度突变（从静止到运动）
3. 分析相邻采样点的时间间隔
4. 统计延迟分布和采样性能

使用方法：
1. 运行程序，保持编码器静止
2. 快速转动编码器（不需要按键！）
3. 程序自动检测运动并计算延迟
4. 重复几次后按 Ctrl+C 查看统计
"""

import minimalmodbus
import time
import numpy as np
from collections import deque
import sys
import termios
import tty
import select
import threading

# 寄存器地址
REG_ANGLE_HIGH = 0x40
REG_ANGLE_LOW = 0x41
RESOLUTION = 65536


class LatencyTester:
    """编码器延迟测试器（自动检测版）"""
    
    def __init__(self, port, slave_id=1, baudrate=115200):
        self.port = port
        self.slave_id = slave_id
        self.baudrate = baudrate
        self.instrument = None
        
        # 数据存储
        self.angle_history = deque(maxlen=200)  # 最近200个角度值
        self.time_history = deque(maxlen=200)   # 对应的时间戳
        
        # 运动检测
        self.motion_events = []  # 检测到的运动事件
        self.sample_intervals = []  # 采样间隔
        self.motion_detected_count = 0
        
        self.running = False
        self.last_angle = 0
        self.last_time = 0
        self.is_moving = False
        self.stable_count = 0
        
        # 阈值设置
        self.change_threshold = 2.0  # 单次采样角度变化阈值（度/sample）
        self.stable_threshold = 0.5  # 静止判定阈值（度/sample）
        self.stable_samples = 5  # 连续多少次静止才认为真正静止
        
    def connect(self):
        """连接编码器"""
        self.instrument = minimalmodbus.Instrument(self.port, self.slave_id)
        self.instrument.serial.baudrate = self.baudrate
        self.instrument.serial.bytesize = 8
        self.instrument.serial.parity = minimalmodbus.serial.PARITY_NONE
        self.instrument.serial.stopbits = 1
        self.instrument.serial.timeout = 0.5
        self.instrument.mode = minimalmodbus.MODE_RTU
        self.instrument.clear_buffers_before_each_transaction = True
        
        # 测试连接
        try:
            self.read_angle()
            print(f"✅ 编码器连接成功: {self.port}")
            return True
        except Exception as e:
            print(f"❌ 编码器连接失败: {e}")
            return False
    
    def read_angle(self):
        """读取角度（批量读取）"""
        regs = self.instrument.read_registers(REG_ANGLE_HIGH, 2, 3)
        raw = (regs[0] << 16) | regs[1]
        angle = (raw / RESOLUTION) * 360.0 * 1.6 % 360.0
        return angle
    
    def detect_motion(self, angle, current_time):
        """
        自动检测运动状态变化
        
        Returns:
            motion_started: 是否刚开始运动
            angle_change: 角度变化量
        """
        if self.last_time == 0:
            self.last_angle = angle
            self.last_time = current_time
            return False, 0
        
        # 计算角度变化
        angle_change = abs(angle - self.last_angle)
        
        # 处理0/360边界跨越
        if angle_change > 180:
            angle_change = 360 - angle_change
        
        # 采样间隔
        time_interval = current_time - self.last_time
        self.sample_intervals.append(time_interval * 1000)  # 转为ms
        
        motion_started = False
        
        # 判断是否在运动
        if angle_change > self.change_threshold:
            # 运动中
            if not self.is_moving:
                # 从静止变为运动！
                motion_started = True
                self.is_moving = True
                self.stable_count = 0
                
                # 记录运动事件
                event = {
                    'time': current_time,
                    'angle': angle,
                    'change': angle_change,
                    'interval': time_interval * 1000  # ms
                }
                self.motion_events.append(event)
                
                self.motion_detected_count += 1
                
                print(f"\n🔥 运动 #{self.motion_detected_count}!")
                print(f"   时刻: {current_time:.6f}")
                print(f"   角度变化: {angle_change:.2f}°")
                print(f"   采样间隔: {time_interval*1000:.2f} ms")
                print(f"   → 延迟上界: {time_interval*1000:.2f} ms")
        
        elif angle_change < self.stable_threshold:
            # 接近静止
            self.stable_count += 1
            if self.stable_count >= self.stable_samples:
                if self.is_moving:
                    # 从运动变为静止
                    self.is_moving = False
                    print(f"   ✓ 已静止")
        else:
            # 中间状态，重置计数
            self.stable_count = 0
        
        # 更新上一次的值
        self.last_angle = angle
        self.last_time = current_time
        
        return motion_started, angle_change
    
    def sampling_loop(self):
        """采样循环（在主线程中运行）"""
        print("\n🚀 开始采样...")
        print("=" * 60)
        print("操作说明：")
        print("  1. 保持编码器静止")
        print("  2. 快速转动编码器（不需要按键！）")
        print("  3. 程序会自动检测运动并计算延迟")
        print("  4. 重复几次后按 Ctrl+C 查看统计")
        print("=" * 60)
        print("\n等待编码器运动...")
        
        sample_count = 0
        start_time = time.time()
        last_print_time = start_time
        
        try:
            while self.running:
                # 读取角度
                current_time = time.time()
                angle = self.read_angle()
                
                # 保存到历史
                self.angle_history.append(angle)
                self.time_history.append(current_time)
                
                sample_count += 1
                
                # 自动检测运动
                motion_started, angle_change = self.detect_motion(angle, current_time)
                
                # 每秒更新一次状态显示
                if current_time - last_print_time >= 1.0:
                    elapsed = time.time() - start_time
                    hz = sample_count / elapsed
                    avg_interval = np.mean(self.sample_intervals[-100:]) if self.sample_intervals else 0
                    
                    status = "运动中..." if self.is_moving else "静止"
                    print(f"\r采样率: {hz:.1f} Hz | 角度: {angle:7.2f}° | "
                          f"状态: {status:8s} | 检测到运动: {self.motion_detected_count} 次 | "
                          f"平均间隔: {avg_interval:.2f} ms", 
                          end="", flush=True)
                    last_print_time = current_time
                
        except KeyboardInterrupt:
            pass
    
    
    def run(self):
        """运行测试"""
        if not self.connect():
            return
        
        self.running = True
        
        # 主线程进行采样
        self.sampling_loop()
        
        # 显示统计结果
        self.show_statistics()
    
    def show_statistics(self):
        """显示统计结果"""
        print("\n\n" + "=" * 60)
        print("📊 延迟和性能统计")
        print("=" * 60)
        
        if not self.motion_events:
            print("⚠️  没有检测到任何运动")
            print("   提示: 请快速转动编码器")
            return
        
        # 采样间隔统计
        if self.sample_intervals:
            intervals = np.array(self.sample_intervals)
            print("\n📈 采样性能:")
            print(f"  总采样次数: {len(intervals)}")
            print(f"  平均采样间隔: {np.mean(intervals):.2f} ms")
            print(f"  中位采样间隔: {np.median(intervals):.2f} ms")
            print(f"  实际采样率: {1000/np.mean(intervals):.1f} Hz")
            print(f"  最快间隔: {np.min(intervals):.2f} ms")
            print(f"  最慢间隔: {np.max(intervals):.2f} ms")
            print(f"  标准差: {np.std(intervals):.2f} ms")
        
        # 运动检测统计
        print(f"\n🔥 运动检测:")
        print(f"  检测到运动次数: {len(self.motion_events)}")
        
        if self.motion_events:
            delays = [e['interval'] for e in self.motion_events]
            changes = [e['change'] for e in self.motion_events]
            
            print(f"\n⏱️  响应延迟（采样间隔 = 延迟上界）:")
            print(f"  平均延迟上界: {np.mean(delays):.2f} ms")
            print(f"  中位延迟上界: {np.median(delays):.2f} ms")
            print(f"  最小延迟上界: {np.min(delays):.2f} ms")
            print(f"  最大延迟上界: {np.max(delays):.2f} ms")
            print(f"  标准差: {np.std(delays):.2f} ms")
            
            print(f"\n📐 运动幅度:")
            print(f"  平均角度变化: {np.mean(changes):.2f}°")
            print(f"  最大角度变化: {np.max(changes):.2f}°")
            
            print(f"\n详细数据:")
            for i, event in enumerate(self.motion_events, 1):
                print(f"  运动 {i}: 延迟≤{event['interval']:.2f}ms, "
                      f"变化={event['change']:.2f}°")
        
        print("\n💡 说明:")
        print("  - 延迟上界 = 两次采样之间的时间间隔")
        print("  - 实际延迟可能在 0 到延迟上界之间")
        print("  - 采样率越高，延迟上界越小")
        
        print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="编码器延迟测试工具")
    parser.add_argument("port", nargs="?", default="/dev/ttyUSB0", help="串口路径")
    parser.add_argument("--baudrate", "-b", type=int, default=115200, help="波特率")
    parser.add_argument("--slave-id", "-s", type=int, default=1, help="从机ID")
    parser.add_argument("--threshold", "-t", type=float, default=2.0, 
                       help="单次采样角度变化阈值（度），默认2.0")
    parser.add_argument("--stable-threshold", type=float, default=0.5,
                       help="静止判定阈值（度），默认0.5")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("编码器端到端延迟测试工具")
    print("=" * 60)
    print(f"串口: {args.port}")
    print(f"波特率: {args.baudrate}")
    print(f"运动检测阈值: {args.threshold}°")
    print(f"静止判定阈值: {args.stable_threshold}°")
    
    tester = LatencyTester(
        args.port, 
        slave_id=args.slave_id, 
        baudrate=args.baudrate
    )
    tester.change_threshold = args.threshold
    tester.stable_threshold = args.stable_threshold
    
    tester.run()


if __name__ == "__main__":
    main()


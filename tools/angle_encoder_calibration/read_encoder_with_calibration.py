#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
磁编码器读取（带自动校准功能）
----------------------------------
✔ 参考 t_265_action_reader.py 的校准方式
✔ 将当前位置设为零点（而不是扫描找最小值）
✔ 正确处理角度跨 0/360 的环绕问题
✔ 使用累积追踪来准确计算相对角度
✔ 自动生成 encoder_config.json
✔ 实时可视化角度变化曲线
"""

import minimalmodbus
import time
import json
import os
import sys
import threading
from collections import deque
import numpy as np

# 配置文件路径
CONFIG_FILE = "encoder_config.json"

# 寄存器地址定义
REG_ANGLE_HIGH = 0x40
REG_ANGLE_LOW  = 0x41

RESOLUTION = 65536  # 根据你的应用选择的分辨率


# -------------------------
# 读取 & 写入 配置文件
# -------------------------
def load_config():
    """加载配置文件"""
    if not os.path.exists(CONFIG_FILE):
        return {"angle_zero": 0.0, "calibrated": False}

    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def save_config(cfg):
    """写入配置文件"""
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"💾 已保存配置到 {CONFIG_FILE}")


# -------------------------
# 角度读取
# -------------------------
def setup_encoder(port, slave_id=1, baudrate=115200):
    """设置编码器连接（使用与 test_encoder.py 相同的配置）"""
    instrument = minimalmodbus.Instrument(port, slave_id)
    instrument.serial.baudrate = baudrate
    instrument.serial.bytesize = 8
    instrument.serial.parity = minimalmodbus.serial.PARITY_NONE
    instrument.serial.stopbits = 1
    instrument.serial.timeout = 0.5
    instrument.mode = minimalmodbus.MODE_RTU
    instrument.clear_buffers_before_each_transaction = True
    return instrument


def read_raw_angle(instrument):
    """读取原始角度值（使用批量读取，与 test_encoder.py 相同的方式）"""
    try:
        # 批量读取2个连续寄存器 (0x40, 0x41) - 与 test_encoder.py 相同
        regs = instrument.read_registers(REG_ANGLE_HIGH, 2, 3)
        raw = (regs[0] << 16) | regs[1]
        # 原始角度计算（* 2.0 系数）
        deg = (raw / RESOLUTION) * 360.0 * 2.0 % 360.0
        return deg, True
    except Exception as e:
        print("❌ 读取角度失败:", e)
        return 0.0, False


# -------------------------
# 角度校准（参考 t_265_action_reader.py）
# -------------------------
def calibrate(instrument, print_info=True):
    """
    校准角度零点（参考 t_265_action_reader.py 的方式）
    
    将当前位置设为零点（闭合/初始位置）
    之后读取的角度将是相对于此零点的角度
    
    Returns:
        校准时的原始角度值
    """
    try:
        ang_raw, ok = read_raw_angle(instrument)
        if not ok:
            print("❌ 读取角度失败，无法校准")
            return 0.0
        
        cfg = {
            "angle_zero": ang_raw,
            "calibrated": True
        }
        save_config(cfg)
        
        if print_info:
            print(f"✅ 零点已设置")
            print(f"   原始角度: {ang_raw:.2f}°")
            print(f"   之后的角度将相对于此零点计算")
        
        return ang_raw
    except Exception as e:
        print(f"❌ 校准失败: {e}")
        return 0.0


# -------------------------
# 角度修正（参考 t_265_action_reader.py）
# -------------------------
def get_calibrated_angle(raw_angle, angle_zero, last_raw_angle, accumulated_turns, scale=0.5):
    """
    计算校准后的角度（参考 t_265_action_reader.py 的逻辑）
    
    处理0~360度的环绕问题：
    - 当角度从359°变到1°时，实际是+2°，不是-358°
    - 当角度从1°变到359°时，实际是-2°，不是+358°
    
    使用累积追踪来正确处理跨越0/360边界的情况
    
    Args:
        raw_angle: 当前原始角度 (度), 范围 0~360
        angle_zero: 零点时的原始角度 (度)
        last_raw_angle: 上一次的原始角度 (度)
        accumulated_turns: 累积的圈数（需要作为列表传入以便修改）
        scale: 角度缩放系数，默认0.5（缩减一半）
        
    Returns:
        (calibrated_angle, new_last_raw_angle, new_accumulated_turns): 
        校准后的角度、新的上次角度、新的累积圈数
    """
    # 检测是否跨越了 0/360 边界
    delta = raw_angle - last_raw_angle
    
    # 如果变化超过180度，说明发生了环绕
    if delta > 180:
        # 从大角度跳到小角度，例如 350° -> 10°，实际是逆时针
        # delta = 10 - 350 = -340，但显示为正，所以实际减少了一圈
        accumulated_turns[0] -= 1
    elif delta < -180:
        # 从小角度跳到大角度，例如 10° -> 350°，实际是顺时针
        # delta = 350 - 10 = 340，但显示为负，所以实际增加了一圈
        accumulated_turns[0] += 1
    
    # 计算相对于零点的角度变化
    angle_diff = raw_angle - angle_zero
    
    # 加上累积的圈数
    total_diff = accumulated_turns[0] * 360.0 + angle_diff
    
    # 反转方向并应用缩放系数
    # 原来增加的现在变成减少，原来减少的现在变成增加
    # 同时缩放为原来的 scale 倍（默认0.5，即一半）
    calibrated_angle = -total_diff * scale
    
    return calibrated_angle, raw_angle, accumulated_turns[0]


# -------------------------
# 实时可视化（多线程分离采样和绘图）
# -------------------------
class AnglePlotter:
    """实时角度曲线绘制器（多线程版本，采样和绘图分离）"""
    
    def __init__(self, window_size=200, y_range=(0, 90)):
        """
        初始化绘图器
        
        Args:
            window_size: 显示的数据点数量（窗口大小）
            y_range: 校准后角度的y轴固定范围（度），元组(min, max)
        """
        self.window_size = window_size
        self.y_range = y_range  # y轴固定范围
        self.times = deque(maxlen=window_size)
        self.angles = deque(maxlen=window_size)
        self.raw_angles = deque(maxlen=window_size)
        
        self.start_time = time.time()
        self.running = False
        self.plot_thread = None
        self.data_lock = threading.Lock()
        
        # matplotlib 对象
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.line1 = None
        self.line2 = None
        self.background1 = None
        self.background2 = None
        
    def start(self):
        """启动绘图（在独立线程中）"""
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            
            self.running = True
            
            # 在独立线程中运行绘图
            self.plot_thread = threading.Thread(target=self._plot_loop, daemon=True)
            self.plot_thread.start()
            
            # 等待绘图窗口初始化
            time.sleep(0.5)
            
            return True
        except Exception as e:
            print(f"⚠️  无法启动绘图: {e}")
            return False
    
    def _plot_loop(self):
        """绘图循环（在独立线程中运行）"""
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            
            # 创建图形
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(14, 10))
            self.fig.suptitle('夹爪角度实时监测 - 最高频率模式', fontsize=18, fontweight='bold')
            
            # 校准后角度 - y轴固定0-90度
            self.ax1.set_title('校准后角度（相对于零点，正值=张开）', fontsize=14)
            self.ax1.set_xlabel('时间 (秒)', fontsize=12)
            self.ax1.set_ylabel('角度 (度)', fontsize=12)
            self.ax1.grid(True, alpha=0.3, linestyle='--')
            
            # 禁用自动缩放，固定y轴范围
            self.ax1.set_ylim(self.y_range[0], self.y_range[1])
            self.ax1.set_autoscale_on(False)  # 关键：禁用自动缩放
            self.ax1.autoscale(enable=False, axis='y')  # 明确禁用y轴自动缩放
            
            self.line1, = self.ax1.plot([], [], 'b-', linewidth=3, label='校准后角度')
            self.ax1.legend(fontsize=11)
            
            # 原始角度（y轴自动调整）
            self.ax2.set_title('原始角度（编码器绝对值）', fontsize=14)
            self.ax2.set_xlabel('时间 (秒)', fontsize=12)
            self.ax2.set_ylabel('角度 (度)', fontsize=12)
            self.ax2.grid(True, alpha=0.3, linestyle='--')
            self.line2, = self.ax2.plot([], [], 'r-', linewidth=3, label='原始角度')
            self.ax2.legend(fontsize=11)
            
            plt.tight_layout()
            plt.ion()
            plt.show()
            
            # 绘图更新循环（约30 FPS，不影响采样）
            while self.running:
                self._redraw_in_thread()
                time.sleep(0.033)  # 约30 FPS
            
        except Exception as e:
            print(f"绘图线程错误: {e}")
    
    def update(self, angle_calibrated, angle_raw):
        """更新数据点（从主采样线程调用，极快）"""
        if not self.running:
            return
        
        current_time = time.time() - self.start_time
        
        # 使用锁保护数据（避免采样线程和绘图线程冲突）
        with self.data_lock:
            self.times.append(current_time)
            self.angles.append(angle_calibrated)
            self.raw_angles.append(angle_raw)
        
        # 不在这里绘图！绘图由独立线程负责
    
    def _redraw_in_thread(self):
        """重绘图形（在绘图线程中调用）"""
        if not self.running or self.fig is None:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # 复制数据（快速锁定）
            with self.data_lock:
                times_list = list(self.times)
                angles_list = list(self.angles)
                raw_angles_list = list(self.raw_angles)
            
            if len(times_list) < 2:
                return
            
            # 更新数据（无锁，绘图线程独占）
            self.line1.set_data(times_list, angles_list)
            self.line2.set_data(times_list, raw_angles_list)
            
            # 更新校准后角度图表 - 只更新x轴，y轴保持固定
            # 手动设置x轴范围
            if times_list[-1] > times_list[0]:
                self.ax1.set_xlim(times_list[0], times_list[-1])
            
            # 强制设置y轴固定范围（绝对不变）
            self.ax1.set_ylim(self.y_range[0], self.y_range[1])
            self.ax1.autoscale(enable=False)  # 确保禁用自动缩放
            
            # 原始角度的 y 轴自动调整（根据当前200帧）
            raw_angles_array = np.array(raw_angles_list)
            y2_min, y2_max = np.min(raw_angles_array), np.max(raw_angles_array)
            y2_range = y2_max - y2_min
            y2_margin = max(y2_range * 0.1, 5)
            self.ax2.set_xlim(times_list[0], times_list[-1])
            self.ax2.set_ylim(y2_min - y2_margin, y2_max + y2_margin)
            
            # 刷新显示
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            pass
    
    def stop(self):
        """停止绘图"""
        self.running = False
        
        # 等待绘图线程结束
        if self.plot_thread and self.plot_thread.is_alive():
            self.plot_thread.join(timeout=1.0)
        
        try:
            if self.fig:
                import matplotlib.pyplot as plt
                plt.ioff()
                plt.close(self.fig)
        except:
            pass


# -------------------------
# 连续读取
# -------------------------
def read_continuous(port, slave_id=1, baudrate=115200, freq=0, force_recalibrate=False, enable_plot=True):
    """
    连续读取角度
    
    Args:
        port: 串口路径
        slave_id: 从机ID
        baudrate: 波特率
        freq: 读取频率 (Hz)，0 表示最高频率（不延迟）
        force_recalibrate: 是否强制重新校准（忽略已保存的校准值）
        enable_plot: 是否启用实时曲线绘制
    """
    # 连接编码器
    print("🔌 正在连接编码器...")
    instrument = setup_encoder(port, slave_id, baudrate)
    
    # 测试读取
    test_angle, ok = read_raw_angle(instrument)
    if not ok:
        print("❌ 无法读取编码器，请检查连接")
        return
    print(f"✅ 编码器连接成功，当前角度: {test_angle:.2f}°")
    
    # 加载配置文件
    cfg = load_config()
    calibrated = cfg.get("calibrated", False) and not force_recalibrate
    
    # 校准流程
    print("\n" + "=" * 60)
    print("📐 角度校准")
    print("=" * 60)
    
    if calibrated:
        angle_zero = cfg.get("angle_zero", 0.0)
        print(f"📌 已加载之前的校准零点: {angle_zero:.2f}°")
        print("\n请确认：")
        print("  1. 夹爪已调整到您想要的初始位置（零点）")
        print("  2. 如果位置不对，请输入 'r' 重新校准")
        print("  3. 如果位置正确，直接按 Enter 继续")
        
        user_input = input("\n您的选择 (Enter=使用已保存的零点, r=重新校准): ").strip().lower()
        
        if user_input == 'r' or user_input == 'recalibrate':
            calibrated = False
            force_recalibrate = True
    
    if not calibrated or force_recalibrate:
        print("\n请按以下步骤操作：")
        print("  1. 调整夹爪到您想要的初始位置（零点）")
        print("  2. 调整好后，按 Enter 键查看当前角度并确认")
        
        # 等待用户调整好位置
        input("\n调整好夹爪位置后，按 Enter 键继续...")
        
        # 读取并显示当前角度
        print("\n正在读取当前角度...")
        current_angle, ok = read_raw_angle(instrument)
        if not ok:
            print("❌ 无法读取角度，请检查连接")
            return
        
        print(f"📊 当前角度: {current_angle:.2f}°")
        print("\n请确认：")
        print("  - 如果这个位置就是您想要的零点，按 Enter 进行校准")
        print("  - 如果需要重新调整，按 Ctrl+C 退出后重新运行程序")
        
        # 等待用户确认
        try:
            input("\n确认位置后，按 Enter 键进行校准...")
        except KeyboardInterrupt:
            print("\n\n⏹️ 用户取消，退出")
            return
        
        # 进行校准
        angle_zero = calibrate(instrument, print_info=True)
        if angle_zero == 0.0:
            print("❌ 校准失败，退出")
            return
    
    # 累积追踪变量（用于处理0/360度环绕）
    last_raw_angle = angle_zero  # 初始化为零点角度
    accumulated_turns = [0]  # 使用列表以便在函数中修改

    # 初始化绘图器
    plotter = None
    if enable_plot:
        plotter = AnglePlotter(window_size=200)
        plot_success = plotter.start()
        if not plot_success:
            plotter = None
            print("   继续以纯文本模式运行...")

    # 连续读取
    print("\n" + "=" * 60)
    print("🚀 开始读取角度 (Ctrl+C 结束)")
    print("=" * 60)
    print("说明：")
    print("  - 原始角度: 编码器的绝对角度值")
    print("  - 校准后角度: 相对于零点的角度（0°表示在零点位置）")
    print("  - 圈数: 相对于零点的累积圈数（+表示顺时针，-表示逆时针）")
    print("  - 采样率: 实际采样频率")
    if plotter:
        print("  - 实时曲线: 显示在独立窗口中")
    if freq == 0:
        print("  ⚡ 最高频率模式：无延迟，以最快速度读取")
    else:
        print(f"  🎯 目标频率: {freq} Hz")
    print("-" * 60)

    # 计算延迟
    interval = 0 if freq == 0 else (1.0 / freq)
    
    # 统计采样率
    sample_count = 0
    start_time = time.time()
    last_print_time = start_time

    try:
        while True:
            ang_raw, ok = read_raw_angle(instrument)
            if not ok:
                print("\n❌ 读取失败")
                continue

            sample_count += 1

            # 使用新的校准逻辑（处理0/360度环绕）
            ang_cal, last_raw_angle, accumulated_turns[0] = get_calibrated_angle(
                ang_raw, angle_zero, last_raw_angle, accumulated_turns
            )

            # 更新绘图（每一帧都添加数据，精准捕获所有变化）
            if plotter:
                plotter.update(ang_cal, ang_raw)

            # 计算当前采样率
            current_time = time.time()
            elapsed = current_time - start_time
            hz = sample_count / elapsed if elapsed > 0 else 0

            # 打印数据（每 10 次采样更新一次终端显示，减少终端刷新开销）
            if sample_count % 10 == 0:
                print(
                    f"\r原始角度: {ang_raw:7.2f}° | 校准后角度: {ang_cal:7.2f}° | "
                    f"圈数: {accumulated_turns[0]:+3d} | 采样率: {hz:6.1f} Hz",
                    end="", flush=True
                )

            # 延迟控制
            if interval > 0:
                time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n⏹️ 用户终止")
        # 显示最终统计
        final_elapsed = time.time() - start_time
        final_hz = sample_count / final_elapsed if final_elapsed > 0 else 0
        print(f"📊 总采样: {sample_count} 次")
        print(f"⏱️  运行时间: {final_elapsed:.1f} 秒")
        print(f"📈 平均采样率: {final_hz:.1f} Hz")
    finally:
        # 停止绘图
        if plotter:
            plotter.stop()


# -------------------------
# 主程序
# -------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="磁编码器读取（带自动校准功能）")
    parser.add_argument("port", type=str, nargs="?", help="串口路径 (例如: /dev/ttyUSB0)")
    parser.add_argument("--baudrate", "-b", type=int, default=115200, help="波特率 (默认: 115200)")
    parser.add_argument("--slave-id", "-s", type=int, default=1, help="从机ID (默认: 1)")
    parser.add_argument("--freq", "-f", type=float, default=0, 
                        help="读取频率 Hz (默认: 0 = 最高频率)")
    parser.add_argument("--recalibrate", "-r", action="store_true", 
                        help="强制重新校准（忽略已保存的校准值）")
    parser.add_argument("--no-plot", action="store_true",
                        help="禁用实时曲线绘制（纯文本模式）")
    parser.add_argument("--max-speed", "-m", action="store_true",
                        help="最高速度模式（等同于 --freq 0 --no-plot）")
    
    args = parser.parse_args()
    
    if not args.port:
        print("用法: python read_encoder_with_calibration.py <串口> [选项]")
        print("示例: python read_encoder_with_calibration.py /dev/ttyUSB0")
        print("      python read_encoder_with_calibration.py /dev/ttyUSB0 --freq 30")
        print("      python read_encoder_with_calibration.py /dev/ttyUSB0 --max-speed")
        print("      python read_encoder_with_calibration.py /dev/ttyUSB0 --recalibrate")
        print("      python read_encoder_with_calibration.py /dev/ttyUSB0 --no-plot")
        return
    
    # 最高速度模式
    if args.max_speed:
        args.freq = 0
        args.no_plot = True
        print("⚡ 最高速度模式：禁用延迟和绘图，以达到最高采样率")
    
    read_continuous(args.port, args.slave_id, args.baudrate, args.freq, 
                   args.recalibrate, enable_plot=not args.no_plot)


if __name__ == "__main__":
    main()

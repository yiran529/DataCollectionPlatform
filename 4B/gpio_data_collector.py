#!/usr/bin/env python3
"""
树莓派4B GPIO数据收集控制器

通过按钮控制数据收集的开始和结束，RGB LED显示状态

状态流程:
1. 程序启动 → 蓝色闪烁（自动初始化）
2. 初始化完成 → 红色（等待录制）
3. 按钮按下 → 绿色（正在录制）
4. 按钮按下 → 绿色闪烁（保存中）→ 红色（等待）
5. 重复3-4...

GPIO连接:
- 按钮: 默认GPIO18 (BCM编号，物理引脚12)，可通过--button参数修改
  - COM(公共端) → GND (如引脚9)
  - NO(常开触点) → GPIO引脚 (如引脚12=GPIO18)
  - 注意: 按钮初始状态为LOW，按下时变为HIGH
- RGB LED:
  - 红色: GPIO22 (BCM编号，物理引脚15)
  - 绿色: GPIO27 (BCM编号，物理引脚13)
  - 蓝色: GPIO23 (BCM编号，物理引脚16)

使用方式:
    sudo python gpio_data_collector.py --keyboard
"""

import sys
import os
import time
import threading
import signal
import select
import termios
import tty
import glob
import subprocess
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 自动检测平台并导入对应的GPIO库
# 支持树莓派（RPi.GPIO）和 Jetson Xavier（Jetson.GPIO）
try:
    # 优先尝试导入 Jetson.GPIO（用于 Nvidia Jetson 平台）
    import Jetson.GPIO as GPIO
    HAS_GPIO = True
    PLATFORM = "Jetson"
    print("✓ 检测到 Jetson 平台，使用 Jetson.GPIO")
except ImportError:
    try:
        # 回退到 RPi.GPIO（用于树莓派平台）
        import RPi.GPIO as GPIO
        HAS_GPIO = True
        PLATFORM = "RaspberryPi"
        print("✓ 检测到树莓派平台，使用 RPi.GPIO")
    except ImportError:
        HAS_GPIO = False
        PLATFORM = "Simulation"
        print("⚠️ 未检测到 GPIO 库，使用模拟模式")

try:
    import cv2
    import numpy as np
    import h5py
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    print(f"⚠️ 缺少依赖: {e}")

# TurboJPEG - 比OpenCV快2-5倍
try:
    from turbojpeg import TurboJPEG, TJPF_BGR
    TURBO_JPEG = TurboJPEG()
    HAS_TURBOJPEG = True
    print("✓ TurboJPEG 已启用（高速压缩）")
except ImportError:
    TURBO_JPEG = None
    HAS_TURBOJPEG = False

from data_coll.sync_data_collector import SyncDataCollector, SyncFrame


class State(Enum):
    """系统状态"""
    INIT = "init"           # 蓝色闪烁 - 初始化中
    IDLE = "idle"           # 红色常亮 - 等待录制
    RECORDING = "recording" # 绿色常亮 - 正在录制
    SAVING = "saving"       # 绿色闪烁 - 正在保存


class LEDColor:
    """LED颜色定义"""
    OFF = (False, False, False)
    RED = (True, False, False)
    GREEN = (False, True, False)
    BLUE = (False, False, True)
    YELLOW = (True, True, False)
    CYAN = (False, True, True)
    MAGENTA = (True, False, True)
    WHITE = (True, True, True)


def find_usb_drive() -> Optional[str]:
    """自动检测U盘挂载路径"""
    # 常见的U盘挂载位置
    search_paths = [
        "/media/*/*",      # Ubuntu/Debian 自动挂载
        "/media/*",        # 备选
        "/mnt/*",          # 手动挂载
        "/run/media/*/*",  # Arch/Fedora
    ]
    
    for pattern in search_paths:
        matches = glob.glob(pattern)
        for path in matches:
            if os.path.isdir(path) and os.access(path, os.W_OK):
                # 检查是否是可移动设备（通过 /proc/mounts）
                try:
                    with open('/proc/mounts', 'r') as f:
                        for line in f:
                            parts = line.split()
                            if len(parts) >= 2 and parts[1] == path:
                                # 检查是否是 USB 设备
                                device = parts[0]
                                if 'sd' in device or 'usb' in device.lower():
                                    print(f"[U盘] 检测到: {path}")
                                    return path
                except:
                    pass
                
                # 如果无法确定，但路径可写，也尝试使用
                if os.path.ismount(path):
                    print(f"[U盘] 检测到挂载点: {path}")
                    return path
    
    # 最后检查 /media 下是否有任何可写目录
    for pattern in ["/media/*/*", "/media/*"]:
        matches = glob.glob(pattern)
        for path in matches:
            if os.path.isdir(path) and os.access(path, os.W_OK):
                print(f"[U盘] 使用目录: {path}")
                return path
    
    return None


def fast_save_to_hdf5(data: List[SyncFrame], output_dir: str,
                      jpeg_quality: int = 80, n_workers: int = 4) -> Optional[str]:
    """
    超优化HDF5保存：流水线压缩+写入，最大批次，减少转换开销
    
    优化点：
    1. 并行JPEG压缩（TurboJPEG优先）
    2. 超大批次写入（减少I/O次数）
    3. 预分配数据集
    4. 减少numpy转换开销
    5. 使用最新HDF5库版本
    """
    if not data:
        print("❌ 无数据")
        return None
    
    # 生成文件名
    prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_data.h5"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    n_frames = len(data)
    encoder_name = "TurboJPEG" if HAS_TURBOJPEG else "OpenCV"
    print(f"保存: {filepath}")
    print(f"  帧数: {n_frames}, 编码器: {encoder_name}, 线程: {n_workers}")
    
    start_time = time.time()
    
    # 步骤1: 并行压缩所有帧到内存
    def encode_frame(idx: int):
        """压缩单帧到内存"""
        frame = data[idx]
        if HAS_TURBOJPEG:
            stereo_jpg = TURBO_JPEG.encode(frame.stereo, quality=jpeg_quality)
            mono_jpg = TURBO_JPEG.encode(frame.mono, quality=jpeg_quality)
            return idx, stereo_jpg, mono_jpg
        else:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            _, stereo_jpg = cv2.imencode('.jpg', frame.stereo, encode_params)
            _, mono_jpg = cv2.imencode('.jpg', frame.mono, encode_params)
            return idx, stereo_jpg.tobytes(), mono_jpg.tobytes()
    
    compress_start = time.time()
    stereo_jpegs = [None] * n_frames
    mono_jpegs = [None] * n_frames
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for idx, stereo_jpg, mono_jpg in executor.map(encode_frame, range(n_frames)):
            stereo_jpegs[idx] = stereo_jpg
            mono_jpegs[idx] = mono_jpg
    
    compress_time = time.time() - compress_start
    fps_compress = n_frames / compress_time if compress_time > 0 else 0
    print(f"  压缩耗时: {compress_time:.2f}s ({fps_compress:.1f} fps)")
    
    # 步骤2: 优化HDF5写入
    write_start = time.time()
    
    # 计算最优chunk和batch大小
    # chunk大小：根据数据量调整，但不要太小，且不能超过数据大小
    chunk_size = min(500, max(100, n_frames // 4), n_frames)
    # batch大小：尽可能大，减少I/O次数（但不要超过chunk）
    batch_size = min(chunk_size * 2, max(500, n_frames // 2), n_frames)
    
    # 预准备时间戳和角度数组（避免在HDF5中重复创建）
    angles_arr = np.array([fr.angle for fr in data], dtype=np.float32)
    timestamps_arr = np.array([fr.timestamp for fr in data], dtype=np.float64)
    stereo_ts_arr = np.array([fr.stereo_ts for fr in data], dtype=np.float64)
    mono_ts_arr = np.array([fr.mono_ts for fr in data], dtype=np.float64)
    encoder_ts_arr = np.array([fr.encoder_ts for fr in data], dtype=np.float64)
    
    with h5py.File(filepath, 'w', libver='latest', swmr=False) as f:
        # 元数据
        f.attrs['n_frames'] = n_frames
        f.attrs['stereo_shape'] = data[0].stereo.shape
        f.attrs['mono_shape'] = data[0].mono.shape
        f.attrs['jpeg_quality'] = jpeg_quality
        f.attrs['created_at'] = datetime.now().isoformat()
        
        # 创建可变长度数据集，使用优化的chunk
        dt = h5py.special_dtype(vlen=np.uint8)
        
        # 使用更大的chunk减少I/O
        stereo_ds = f.create_dataset(
            'stereo_jpeg',
            shape=(n_frames,),
            dtype=dt,
            chunks=(chunk_size,),
            compression=None,
            shuffle=False,  # 不shuffle（已经是压缩数据）
            fletcher32=False  # 不校验（加快写入）
        )
        mono_ds = f.create_dataset(
            'mono_jpeg',
            shape=(n_frames,),
            dtype=dt,
            chunks=(chunk_size,),
            compression=None,
            shuffle=False,
            fletcher32=False
        )
        
        # 超大批次写入（减少I/O次数）
        # 直接使用bytes，避免不必要的numpy转换
        for i in range(0, n_frames, batch_size):
            end = min(i + batch_size, n_frames)
            # 批量转换（只在需要时转换）
            stereo_batch = [np.frombuffer(s, dtype=np.uint8) if isinstance(s, (bytes, bytearray)) else s 
                          for s in stereo_jpegs[i:end]]
            mono_batch = [np.frombuffer(m, dtype=np.uint8) if isinstance(m, (bytes, bytearray)) else m 
                         for m in mono_jpegs[i:end]]
            stereo_ds[i:end] = stereo_batch
            mono_ds[i:end] = mono_batch
        
        # 一次性写入所有时间戳和角度（使用预准备的数组）
        f.create_dataset(
            'angles',
            data=angles_arr,
            compression=None,
            chunks=(min(2000, n_frames),),
            shuffle=False,
            fletcher32=False
        )
        f.create_dataset(
            'timestamps',
            data=timestamps_arr,
            compression=None,
            chunks=(min(2000, n_frames),),
            shuffle=False,
            fletcher32=False
        )
        f.create_dataset(
            'stereo_timestamps',
            data=stereo_ts_arr,
            compression=None,
            chunks=(min(2000, n_frames),),
            shuffle=False,
            fletcher32=False
        )
        f.create_dataset(
            'mono_timestamps',
            data=mono_ts_arr,
            compression=None,
            chunks=(min(2000, n_frames),),
            shuffle=False,
            fletcher32=False
        )
        f.create_dataset(
            'encoder_timestamps',
            data=encoder_ts_arr,
            compression=None,
            chunks=(min(2000, n_frames),),
            shuffle=False,
            fletcher32=False
        )
    
    write_time = time.time() - write_start
    total_time = time.time() - start_time
    
    file_size = os.path.getsize(filepath) / (1024 * 1024)
    speed = file_size / total_time if total_time > 0 else 0
    
    print(f"  写入耗时: {write_time:.2f}s")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  文件大小: {file_size:.1f}MB")
    print(f"  写入速度: {speed:.1f}MB/s")
    
    # 异步同步（不阻塞）
    def async_sync():
        subprocess.run(['sync'], check=False)
    
    sync_thread = threading.Thread(target=async_sync, daemon=True)
    sync_thread.start()
    
    return filepath


def fast_save_npz(data: List[SyncFrame], output_dir: str,
                  jpeg_quality: int = 80, n_workers: int = 4) -> Optional[str]:
    """
    快速保存：单个NPZ文件（比HDF5快3-5倍）
    """
    if not data:
        print("❌ 无数据")
        return None
    
    prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_data.npz"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    n_frames = len(data)
    encoder_name = "TurboJPEG" if HAS_TURBOJPEG else "OpenCV"
    print(f"保存: {filepath}")
    print(f"  帧数: {n_frames}, 编码器: {encoder_name}, 线程: {n_workers}")
    
    start_time = time.time()
    
    # 并行JPEG压缩
    if HAS_TURBOJPEG:
        def encode_frame(idx: int):
            frame = data[idx]
            return (idx,
                    TURBO_JPEG.encode(frame.stereo, quality=jpeg_quality),
                    TURBO_JPEG.encode(frame.mono, quality=jpeg_quality))
    else:
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        def encode_frame(idx: int):
            frame = data[idx]
            _, s = cv2.imencode('.jpg', frame.stereo, params)
            _, m = cv2.imencode('.jpg', frame.mono, params)
            return idx, s.tobytes(), m.tobytes()
    
    stereo_jpegs = [None] * n_frames
    mono_jpegs = [None] * n_frames
    
    compress_start = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for idx, s, m in executor.map(encode_frame, range(n_frames)):
            stereo_jpegs[idx] = s
            mono_jpegs[idx] = m
    
    compress_time = time.time() - compress_start
    print(f"  压缩耗时: {compress_time:.2f}s ({n_frames/compress_time:.1f} fps)")
    
    # 打包成bytes数组（使用pickle序列化，比HDF5的vlen快很多）
    write_start = time.time()
    
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump({
            'stereo_jpegs': stereo_jpegs,
            'mono_jpegs': mono_jpegs,
            'angles': np.array([fr.angle for fr in data], dtype=np.float32),
            'timestamps': np.array([fr.timestamp for fr in data], dtype=np.float64),
            'stereo_timestamps': np.array([fr.stereo_ts for fr in data], dtype=np.float64),
            'mono_timestamps': np.array([fr.mono_ts for fr in data], dtype=np.float64),
            'encoder_timestamps': np.array([fr.encoder_ts for fr in data], dtype=np.float64),
            'stereo_shape': data[0].stereo.shape,
            'mono_shape': data[0].mono.shape,
            'n_frames': n_frames,
            'jpeg_quality': jpeg_quality,
            'created_at': datetime.now().isoformat()
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    write_time = time.time() - write_start
    total_time = time.time() - start_time
    
    file_size = os.path.getsize(filepath) / (1024 * 1024)
    speed = file_size / total_time if total_time > 0 else 0
    
    print(f"  写入耗时: {write_time:.2f}s")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  文件大小: {file_size:.1f}MB")
    print(f"  写入速度: {speed:.1f}MB/s")
    
    subprocess.run(['sync'], check=False)
    return filepath


class KeyboardListener:
    """键盘监听器"""
    
    def __init__(self, trigger_key: str = '1'):
        self.trigger_key = trigger_key
        self._callback = None
        self._running = False
        self._thread = None
        self._old_settings = None
    
    def start(self, callback):
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print(f"[键盘] 按 '{self.trigger_key}' 键触发操作")
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
    
    def _listen_loop(self):
        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            
            while self._running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch == self.trigger_key:
                        if self._callback:
                            self._callback()
                    elif ch == '\x03':
                        self._running = False
                        os.kill(os.getpid(), signal.SIGINT)
                        break
        except:
            pass
        finally:
            if self._old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)


class GPIOController:
    """GPIO控制器"""
    
    # 三引脚按钮模块: VCC→3.3V(引脚1), GND→GND(引脚6), OUT→GPIO4(引脚7)
    DEFAULT_BUTTON_PIN = 4   # GPIO4, 物理引脚7
    DEFAULT_LED_RED_PIN = 22
    DEFAULT_LED_GREEN_PIN = 27
    DEFAULT_LED_BLUE_PIN = 23
    
    def __init__(self, button_pin: int = None, led_pins: tuple = None,
                 use_keyboard: bool = False, keyboard_key: str = '1'):
        self.button_pin = button_pin or self.DEFAULT_BUTTON_PIN
        self.led_red = led_pins[0] if led_pins else self.DEFAULT_LED_RED_PIN
        self.led_green = led_pins[1] if led_pins else self.DEFAULT_LED_GREEN_PIN
        self.led_blue = led_pins[2] if led_pins else self.DEFAULT_LED_BLUE_PIN
        
        self._button_callback = None
        self._last_trigger_time = 0
        self._debounce_ms = 300
        
        self.use_keyboard = use_keyboard
        self._keyboard = KeyboardListener(keyboard_key) if use_keyboard else None
        
        # 轮询模式（当事件检测失败时使用）
        self._use_polling = False
        self._polling_thread = None
        self._polling_stop = threading.Event()
        self._button_initial_state = None
        
        # 非自锁按钮状态跟踪：检测"按下-松开"完整周期
        self._button_pressed = False  # 是否处于按下状态
        
        if HAS_GPIO:
            self._setup_gpio()
    
    def _setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # 先清理可能存在的旧事件检测
        try:
            GPIO.remove_event_detect(self.button_pin)
        except:
            pass
        
        GPIO.setup(self.led_red, GPIO.OUT)
        GPIO.setup(self.led_green, GPIO.OUT)
        GPIO.setup(self.led_blue, GPIO.OUT)
        # 三引脚按钮模块：松开时OUT输出LOW，按下时OUT输出HIGH
        GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        self.set_led(LEDColor.OFF)
        print(f"[GPIO] LED: R={self.led_red}, G={self.led_green}, B={self.led_blue}")
        print(f"[GPIO] 按钮: GPIO{self.button_pin} (物理引脚7)")
        print(f"[GPIO] 按钮类型: 非自锁按钮（松开=LOW, 按下=HIGH）")
    
    def set_led(self, color: tuple):
        if HAS_GPIO:
            GPIO.output(self.led_red, color[0])
            GPIO.output(self.led_green, color[1])
            GPIO.output(self.led_blue, color[2])
        else:
            names = {LEDColor.OFF: "灭", LEDColor.RED: "红", LEDColor.GREEN: "绿",
                    LEDColor.BLUE: "蓝", LEDColor.YELLOW: "黄"}
            print(f"[LED] → {names.get(color, str(color))}")
    
    def set_button_callback(self, callback):
        self._button_callback = callback
        
        if self.use_keyboard and self._keyboard:
            self._keyboard.start(self._keyboard_handler)
        
        if HAS_GPIO:
            try:
                # 先移除可能存在的旧事件检测
                try:
                    GPIO.remove_event_detect(self.button_pin)
                except RuntimeError:
                    pass  # 如果没有旧检测，忽略错误
                
                # 检测初始状态
                # 三引脚按钮模块：松开=LOW，按下=HIGH
                initial_state = GPIO.input(self.button_pin)
                self._button_pressed = (initial_state == GPIO.HIGH)
                print(f"[GPIO] 按钮初始状态: {'HIGH(按下)' if initial_state == GPIO.HIGH else 'LOW(松开)'}")
                
                # 非自锁按钮：检测双边沿，在松开时触发回调
                # 按下时(LOW→HIGH)：记录按下状态
                # 松开时(HIGH→LOW)：如果之前是按下状态，触发回调
                edge = GPIO.BOTH
                print(f"[GPIO] 按钮检测: 非自锁按钮 - 检测按下-松开周期")
                
                GPIO.add_event_detect(self.button_pin, edge,
                                     callback=self._button_handler,
                                     bouncetime=50)  # 较短的防抖时间用于边沿检测
                print(f"[GPIO] 按钮监听已启动（事件检测模式）")
            except Exception as e:
                print(f"[GPIO] 按钮设置失败: {e}")
                print(f"[GPIO] 自动切换到轮询模式...")
                self._use_polling = True
                self._button_initial_state = initial_state
                self._start_polling()
    
    def _keyboard_handler(self):
        current = time.time() * 1000
        if current - self._last_trigger_time < self._debounce_ms:
            return
        self._last_trigger_time = current
        if self._button_callback:
            self._button_callback()
    
    def _button_handler(self, channel):
        """非自锁按钮处理：检测按下-松开完整周期
        
        三引脚按钮模块逻辑：
        - 松开时：OUT = LOW
        - 按下时：OUT = HIGH
        """
        if not HAS_GPIO:
            return
        
        current_state = GPIO.input(self.button_pin)
        current_time = time.time() * 1000
        
        if current_state == GPIO.HIGH:
            # 按下（上升沿）：记录按下状态
            self._button_pressed = True
            print(f"[GPIO] 按钮按下")
        else:
            # 松开（下降沿）：如果之前是按下状态，触发回调
            if self._button_pressed:
                self._button_pressed = False
                
                # 防抖处理
                if current_time - self._last_trigger_time < self._debounce_ms:
                    print(f"[GPIO] 按钮松开（防抖忽略）")
                    return
                
                self._last_trigger_time = current_time
                print(f"[GPIO] 按钮松开 → 触发信号")
                
        if self._button_callback:
            self._button_callback()
    
    def _start_polling(self):
        """启动轮询模式检测按钮"""
        if not HAS_GPIO or self._use_polling is False:
            return
        
        self._polling_stop.clear()
        self._polling_thread = threading.Thread(
            target=self._polling_loop, daemon=True)
        self._polling_thread.start()
        print(f"[GPIO] 按钮监听已启动（轮询模式）")
    
    def _polling_loop(self):
        """轮询循环检测按钮状态（非自锁按钮：检测按下-松开周期）
        
        三引脚按钮模块逻辑：
        - 松开时：OUT = LOW
        - 按下时：OUT = HIGH
        """
        if not HAS_GPIO:
            return
        
        # 读取初始状态
        last_state = GPIO.input(self.button_pin)
        self._button_pressed = (last_state == GPIO.HIGH)
        last_trigger_time = 0
        
        while not self._polling_stop.is_set():
            try:
                current_state = GPIO.input(self.button_pin)
                current_time = time.time() * 1000
                
                # 检测状态变化
                if current_state != last_state:
                    if current_state == GPIO.HIGH:
                        # 按下（上升沿）：记录按下状态
                        self._button_pressed = True
                        print(f"[GPIO] 按钮按下")
                    else:
                        # 松开（下降沿）：如果之前是按下状态，触发回调
                        if self._button_pressed:
                            self._button_pressed = False
                            
                    # 防抖处理
                            if current_time - last_trigger_time > self._debounce_ms:
                                last_trigger_time = current_time
                                print(f"[GPIO] 按钮松开 → 触发信号")
                                
                        if self._button_callback:
                            self._button_callback()
                            else:
                                print(f"[GPIO] 按钮松开（防抖忽略）")
                        
                        last_state = current_state
                
                time.sleep(0.01)  # 10ms轮询间隔
            except:
                break
    
    def cleanup(self):
        if self._keyboard:
            self._keyboard.stop()
        
        # 停止轮询线程
        if self._polling_thread:
            self._polling_stop.set()
            if self._polling_thread.is_alive():
                self._polling_thread.join(timeout=1)
        
        if HAS_GPIO:
            try:
                GPIO.remove_event_detect(self.button_pin)
            except:
                pass
            self.set_led(LEDColor.OFF)
            GPIO.cleanup()


class GPIODataCollector:
    """GPIO数据收集控制器"""
    
    def __init__(self, config_path: str, gpio: GPIOController = None):
        self.config_path = config_path
        self.gpio = gpio or GPIOController()
        
        self.collector: SyncDataCollector = None
        self.state = State.INIT
        self.running = True
        self.session_count = 0
        
        # U盘路径
        self.usb_path: str = None
        self.save_dir: str = None
        
        # 按钮事件
        self.gpio.set_button_callback(self._on_button_press)
        self._button_event = threading.Event()
        
        # LED闪烁
        self._blink_thread = None
        self._blink_stop = threading.Event()
    
    def _set_state(self, state: State):
        """设置状态"""
        self._stop_blink()
        self.state = state
        
        if state == State.INIT:
            self._start_blink(LEDColor.BLUE, 0.5)
        elif state == State.IDLE:
            self.gpio.set_led(LEDColor.RED)
        elif state == State.RECORDING:
            self.gpio.set_led(LEDColor.GREEN)
        elif state == State.SAVING:
            self._start_blink(LEDColor.GREEN, 0.15)
        
        print(f"[状态] {state.value}")
    
    def _start_blink(self, color: tuple, interval: float):
        self._blink_stop.clear()
        self._blink_thread = threading.Thread(
            target=self._blink_loop, args=(color, interval), daemon=True)
        self._blink_thread.start()
    
    def _stop_blink(self):
        self._blink_stop.set()
        if self._blink_thread:
            self._blink_thread.join(timeout=1)
            self._blink_thread = None
    
    def _blink_loop(self, color: tuple, interval: float):
        on = True
        while not self._blink_stop.is_set():
            self.gpio.set_led(color if on else LEDColor.OFF)
            on = not on
            self._blink_stop.wait(interval)
    
    def _on_button_press(self):
        self._button_event.set()
    
    def _wait_button(self):
        self._button_event.clear()
        self._button_event.wait()
    
    def _detect_usb(self) -> bool:
        """检测U盘"""
        self.usb_path = find_usb_drive()
        if self.usb_path:
            self.save_dir = os.path.join(self.usb_path, "video")
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"[存储] 保存目录: {self.save_dir}")
            return True
        else:
            print("⚠️ 未检测到U盘，使用本地目录")
            self.save_dir = os.path.join(os.path.dirname(__file__), "data")
            os.makedirs(self.save_dir, exist_ok=True)
            return False
    
    def _initialize(self) -> bool:
        """初始化"""
        print("\n" + "=" * 50)
        print("系统初始化...")
        print("=" * 50)
        
        # 检测U盘
        self._detect_usb()
        
        # 初始化数据收集器
        print("\n初始化相机和编码器...")
        self.collector = SyncDataCollector(self.config_path)
        self.collector.start()
        
        if not self.collector.wait_ready(timeout=60):
            print("❌ 初始化超时")
            return False
        
        print("✅ 初始化完成")
        return True
    
    def _start_recording(self):
        """开始录制"""
        self.session_count += 1
        print(f"\n🔴 开始录制 (Session #{self.session_count})")
        return self.collector.start_recording()
    
    def _stop_recording(self):
        """停止录制并保存"""
        data = self.collector.stop_recording()
        
        if data:
            # 重新检测U盘（可能中途插入）
            new_usb = find_usb_drive()
            if new_usb:
                self.save_dir = os.path.join(new_usb, "video")
                os.makedirs(self.save_dir, exist_ok=True)
            
            # 使用优化的HDF5保存（先压缩到内存，再批量写入）
            filepath = fast_save_to_hdf5(
                data, 
                self.save_dir,
                jpeg_quality=80,
                n_workers=4
            )
            
            if filepath:
                print(f"✅ 已保存: {filepath}")
            else:
                print("❌ 保存失败")
        else:
            print("⚠️ 无数据")
    
    def run(self):
        """主循环"""
        print("\n" + "=" * 50)
        print("GPIO数据收集控制器 v2.0")
        print("=" * 50)
        
        try:
            # 启动时自动初始化
            self._set_state(State.INIT)
            
            if not self._initialize():
                print("❌ 初始化失败，退出")
                return
            
            # 初始化完成，进入等待状态
            self._set_state(State.IDLE)
            print("\n等待按钮开始录制...")
            
            while self.running:
                if self.state == State.IDLE:
                    self._wait_button()
                    if not self.running:
                        break
                    
                    self._set_state(State.RECORDING)
                    self._start_recording()
                
                elif self.state == State.RECORDING:
                    self._wait_button()
                    if not self.running:
                        break
                    
                    self._set_state(State.SAVING)
                    self._stop_recording()
                    
                    self._set_state(State.IDLE)
                    print("\n等待下一次录制...")
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n中断")
        finally:
            self.stop()
    
    def stop(self):
        """停止"""
        self.running = False
        self._button_event.set()
        self._stop_blink()
        
        if self.collector:
            if self.state == State.RECORDING:
                self._set_state(State.SAVING)
                self._stop_recording()
            self.collector.stop()
        
        self.gpio.cleanup()
        print("\n已停止")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GPIO数据收集控制器")
    parser.add_argument("--config", "-c", type=str,
                        default="../data_coll/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--keyboard", "-k", action="store_true",
                        help="使用键盘代替按钮")
    parser.add_argument("--key", type=str, default="1",
                        help="触发按键 (默认: 1)")
    parser.add_argument("--button", type=int, default=4,
                        help="按钮GPIO引脚 (BCM编号，默认: 4, 物理引脚7)")
    parser.add_argument("--led-red", type=int, default=22)
    parser.add_argument("--led-green", type=int, default=27)
    parser.add_argument("--led-blue", type=int, default=23)
    args = parser.parse_args()
    
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    
    gpio = GPIOController(
        button_pin=args.button,
        led_pins=(args.led_red, args.led_green, args.led_blue),
        use_keyboard=args.keyboard,
        keyboard_key=args.key
    )
    
    controller = GPIODataCollector(config_path, gpio)
    
    def signal_handler(sig, frame):
        print("\n收到终止信号...")
        controller.running = False
        controller._button_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    controller.run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
按钮功能测试脚本

专门用于测试三引脚按钮模块是否正常工作

接线:
- VCC → 3.3V (物理引脚1)
- GND → GND (物理引脚6)
- OUT → GPIO4 (物理引脚7)
"""

import time
import sys
import os

try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    print("⚠️ RPi.GPIO 未安装")
    print("  安装命令: pip install RPi.GPIO")
    print("  或者: sudo apt install python3-rpi.gpio")
    sys.exit(1)


def check_permissions():
    """检查GPIO权限"""
    if os.geteuid() == 0:
        return True, "root"
    
    import grp
    try:
        gpio_gid = grp.getgrnam('gpio').gr_gid
        user_gids = os.getgroups()
        if gpio_gid in user_gids:
            return True, "gpio_group"
    except (KeyError, AttributeError):
        pass
    
    if os.path.exists('/dev/gpiomem'):
        stat_info = os.stat('/dev/gpiomem')
        mode = stat_info.st_mode & 0o777
        if mode & 0o020:
            return True, "device_permission"
    
    return False, "no_permission"


# 按钮引脚配置
BUTTON_PIN = 4  # BCM编号，物理引脚7


def setup():
    """初始化GPIO"""
    # 检查权限
    has_permission, perm_type = check_permissions()
    if not has_permission:
        print("\n" + "=" * 60)
        print("⚠️ GPIO权限错误")
        print("=" * 60)
        print("错误: No access to /dev/mem")
        print("\n解决方案:")
        print("\n方案1: 使用sudo运行（推荐）")
        print("  sudo python3 test_button.py")
        print("\n方案2: 修复GPIO权限（永久解决）")
        print("  cd /home/cjq/Documents/DataCollectionPlatform/4B")
        print("  sudo bash fix_gpio_permissions.sh")
        print("  然后重新登录或重启: sudo reboot")
        print("=" * 60)
        sys.exit(1)
    
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # 按钮输入（内部下拉）
        # 三引脚按钮模块：松开时OUT输出LOW，按下时OUT输出HIGH
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        
        print("✓ GPIO初始化成功")
        print(f"  按钮引脚: GPIO{BUTTON_PIN} (物理引脚7)")
        print(f"  接线: VCC→3.3V(引脚1), GND→GND(引脚6), OUT→GPIO{BUTTON_PIN}(引脚7)")
        
    except RuntimeError as e:
        if "No access to /dev/mem" in str(e) or "try running as root" in str(e).lower():
            print("\n" + "=" * 60)
            print("⚠️ GPIO权限错误")
            print("=" * 60)
            print(f"错误: {e}")
            print("\n解决方案:")
            print("\n方案1: 使用sudo运行（推荐）")
            print("  sudo python3 test_button.py")
            print("\n方案2: 修复GPIO权限（永久解决）")
            print("  cd /home/cjq/Documents/DataCollectionPlatform/4B")
            print("  sudo bash fix_gpio_permissions.sh")
            print("  然后重新登录或重启: sudo reboot")
            print("=" * 60)
            sys.exit(1)
        else:
            raise


def test_button_basic():
    """测试1: 基本按钮状态检测"""
    print("\n" + "=" * 50)
    print("测试1: 基本按钮状态检测")
    print("=" * 50)
    print("请按下和松开按钮，观察状态变化")
    print("按 Ctrl+C 退出测试\n")
    
    try:
        last_state = None
        press_count = 0
        release_count = 0
        
        print("开始监测按钮状态...")
        print("(松开=LOW, 按下=HIGH)\n")
        
        while True:
            current_state = GPIO.input(BUTTON_PIN)
            
            if current_state != last_state:
                if current_state == GPIO.HIGH:
                    press_count += 1
                    print(f"  [{press_count}] 按钮按下 (HIGH)")
                else:
                    release_count += 1
                    print(f"  [{release_count}] 按钮松开 (LOW)")
                
                last_state = current_state
            
            time.sleep(0.01)  # 10ms轮询间隔
            
    except KeyboardInterrupt:
        print(f"\n\n测试结果:")
        print(f"  按下次数: {press_count}")
        print(f"  松开次数: {release_count}")
        print("\n✓ 基本测试完成")


def test_button_press_count():
    """测试2: 按钮按下计数"""
    print("\n" + "=" * 50)
    print("测试2: 按钮按下计数")
    print("=" * 50)
    print("请按下按钮10次，程序会统计按下次数")
    print("按 Ctrl+C 跳过测试\n")
    
    try:
        press_count = 0
        last_state = GPIO.LOW
        
        print("等待按钮按下...")
        
        while press_count < 10:
            current_state = GPIO.input(BUTTON_PIN)
            
            # 检测上升沿（从LOW到HIGH，表示按下）
            if last_state == GPIO.LOW and current_state == GPIO.HIGH:
                press_count += 1
                print(f"  按下 #{press_count}/10")
                time.sleep(0.2)  # 防抖
            
            last_state = current_state
            time.sleep(0.01)
        
        print(f"\n✓ 成功检测到 {press_count} 次按下")
        
    except KeyboardInterrupt:
        print(f"\n跳过测试（已检测到 {press_count} 次按下）")


def test_button_response_time():
    """测试3: 按钮响应时间测试"""
    print("\n" + "=" * 50)
    print("测试3: 按钮响应时间测试")
    print("=" * 50)
    print("快速按下和松开按钮，测试响应速度")
    print("按 Ctrl+C 退出测试\n")
    
    try:
        last_state = GPIO.LOW
        press_times = []
        release_times = []
        
        print("开始监测...")
        
        while True:
            current_state = GPIO.input(BUTTON_PIN)
            current_time = time.time()
            
            # 检测上升沿（按下）
            if last_state == GPIO.LOW and current_state == GPIO.HIGH:
                press_times.append(current_time)
                if len(press_times) > 1:
                    interval = current_time - press_times[-2]
                    print(f"  按下 #{len(press_times)} (间隔: {interval:.3f}秒)")
                else:
                    print(f"  按下 #1")
            
            # 检测下降沿（松开）
            elif last_state == GPIO.HIGH and current_state == GPIO.LOW:
                release_times.append(current_time)
                if len(release_times) > 0 and len(press_times) > 0:
                    hold_time = current_time - press_times[-1]
                    print(f"  松开 (按住时长: {hold_time:.3f}秒)")
            
            last_state = current_state
            time.sleep(0.001)  # 1ms轮询，提高响应速度
            
    except KeyboardInterrupt:
        if len(press_times) > 1:
            intervals = [press_times[i] - press_times[i-1] for i in range(1, len(press_times))]
            avg_interval = sum(intervals) / len(intervals)
            print(f"\n\n统计结果:")
            print(f"  总按下次数: {len(press_times)}")
            print(f"  平均间隔: {avg_interval:.3f}秒")
        print("\n✓ 响应时间测试完成")


def test_button_continuous_monitor():
    """测试4: 持续监测按钮状态"""
    print("\n" + "=" * 50)
    print("测试4: 持续监测按钮状态")
    print("=" * 50)
    print("实时显示按钮状态，按 Ctrl+C 退出\n")
    
    try:
        while True:
            state = GPIO.input(BUTTON_PIN)
            status = "按下 (HIGH)" if state == GPIO.HIGH else "松开 (LOW)"
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            
            print(f"\r[{timestamp}] 按钮状态: {status}   ", end="", flush=True)
            time.sleep(0.05)  # 50ms更新间隔
            
    except KeyboardInterrupt:
        print("\n\n✓ 监测结束")


def main():
    print("\n" + "=" * 50)
    print("按钮功能测试程序")
    print("=" * 50)
    print(f"""
接线配置:
  VCC → 3.3V (物理引脚1)
  GND → GND (物理引脚6)
  OUT → GPIO{BUTTON_PIN} (物理引脚7)

工作原理:
  - 松开时: OUT输出LOW (低电平)
  - 按下时: OUT输出HIGH (高电平)
  - GPIO配置: 内部下拉 (PUD_DOWN)
""")
    
    setup()
    
    try:
        while True:
            print("\n" + "-" * 50)
            print("选择测试项目:")
            print("  1. 基本按钮状态检测")
            print("  2. 按钮按下计数 (10次)")
            print("  3. 按钮响应时间测试")
            print("  4. 持续监测按钮状态")
            print("  5. 运行所有测试")
            print("  0. 退出")
            print("-" * 50)
            
            choice = input("请选择 (0-5): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                test_button_basic()
            elif choice == '2':
                test_button_press_count()
            elif choice == '3':
                test_button_response_time()
            elif choice == '4':
                test_button_continuous_monitor()
            elif choice == '5':
                test_button_basic()
                time.sleep(1)
                test_button_press_count()
                time.sleep(1)
                test_button_response_time()
            else:
                print("无效选择")
    
    except KeyboardInterrupt:
        print("\n\n程序被中断")
    
    finally:
        GPIO.cleanup()
        print("\n✓ GPIO已清理，程序退出")


if __name__ == "__main__":
    main()


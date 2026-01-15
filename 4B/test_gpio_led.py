#!/usr/bin/env python3
"""
GPIO调试测试程序

分步测试LED和按钮是否正常工作
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
    # 检查是否为root
    if os.geteuid() == 0:
        return True, "root"
    
    # 检查用户是否在gpio组中
    import grp
    try:
        gpio_gid = grp.getgrnam('gpio').gr_gid
        user_gids = os.getgroups()
        if gpio_gid in user_gids:
            return True, "gpio_group"
    except (KeyError, AttributeError):
        pass
    
    # 检查/dev/gpiomem权限
    if os.path.exists('/dev/gpiomem'):
        stat_info = os.stat('/dev/gpiomem')
        mode = stat_info.st_mode & 0o777
        if mode & 0o020:  # 组可写
            return True, "device_permission"
    
    return False, "no_permission"

# 默认引脚配置 (BCM编号)
LED_RED = 22    # 物理引脚 15
LED_GREEN = 27  # 物理引脚 13
LED_BLUE = 23   # 物理引脚 16
BUTTON = 17     # 物理引脚 11


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
        print("  sudo python3 test_gpio_led.py")
        print("\n方案2: 修复GPIO权限（永久解决）")
        print("  cd /home/cjq/Documents/DataCollectionPlatform/4B")
        print("  sudo bash fix_gpio_permissions.sh")
        print("  然后重新登录或重启: sudo reboot")
        print("\n方案3: 临时切换到gpio组（当前会话）")
        print("  newgrp gpio")
        print("  python3 test_gpio_led.py")
        print("=" * 60)
        sys.exit(1)
    
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # LED输出
        GPIO.setup(LED_RED, GPIO.OUT)
        GPIO.setup(LED_GREEN, GPIO.OUT)
        GPIO.setup(LED_BLUE, GPIO.OUT)
        
        # 按钮输入（内部上拉）
        # GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # 初始全灭
        GPIO.output(LED_RED, False)
        GPIO.output(LED_GREEN, False)
        GPIO.output(LED_BLUE, False)
        
        print("✓ GPIO初始化成功")
        
    except RuntimeError as e:
        if "No access to /dev/mem" in str(e) or "try running as root" in str(e).lower():
            print("\n" + "=" * 60)
            print("⚠️ GPIO权限错误")
            print("=" * 60)
            print(f"错误: {e}")
            print("\n解决方案:")
            print("\n方案1: 使用sudo运行（推荐）")
            print("  sudo python3 test_gpio_led.py")
            print("\n方案2: 修复GPIO权限（永久解决）")
            print("  cd /home/cjq/Documents/DataCollectionPlatform/4B")
            print("  sudo bash fix_gpio_permissions.sh")
            print("  然后重新登录或重启: sudo reboot")
            print("\n方案3: 临时切换到gpio组（当前会话）")
            print("  newgrp gpio")
            print("  python3 test_gpio_led.py")
            print("=" * 60)
            sys.exit(1)
        else:
            raise


def test_led_individual():
    """测试1: 单独测试每个颜色"""
    print("\n" + "=" * 40)
    print("测试1: LED单色测试")
    print("=" * 40)
    
    colors = [
        (LED_RED, "红色", "物理引脚15"),
        (LED_GREEN, "绿色", "物理引脚13"),
        (LED_BLUE, "蓝色", "物理引脚16"),
    ]
    
    for pin, name, physical in colors:
        print(f"\n点亮 {name} ({physical})...")
        GPIO.output(pin, True)
        time.sleep(2)
        GPIO.output(pin, False)
        
        result = input(f"  {name}是否亮起? (y/n): ").strip().lower()
        if result != 'y':
            print(f"  ⚠️ {name}可能接线有问题，检查{physical}连接")
    
    print("\n✓ LED单色测试完成")


def test_led_combined():
    """测试2: 组合颜色测试"""
    print("\n" + "=" * 40)
    print("测试2: LED组合颜色测试")
    print("=" * 40)
    
    combinations = [
        ((True, True, False), "黄色 (红+绿)"),
        ((True, False, True), "紫色 (红+蓝)"),
        ((False, True, True), "青色 (绿+蓝)"),
        ((True, True, True), "白色 (全亮)"),
        ((False, False, False), "熄灭"),
    ]
    
    for (r, g, b), name in combinations:
        print(f"\n显示 {name}...")
        GPIO.output(LED_RED, r)
        GPIO.output(LED_GREEN, g)
        GPIO.output(LED_BLUE, b)
        time.sleep(1.5)
    
    print("\n✓ 组合颜色测试完成")


def test_led_blink():
    """测试3: LED闪烁测试"""
    print("\n" + "=" * 40)
    print("测试3: LED闪烁测试 (红→绿→蓝 循环5次)")
    print("=" * 40)
    
    for i in range(5):
        # 红
        GPIO.output(LED_RED, True)
        GPIO.output(LED_GREEN, False)
        GPIO.output(LED_BLUE, False)
        time.sleep(0.3)
        
        # 绿
        GPIO.output(LED_RED, False)
        GPIO.output(LED_GREEN, True)
        GPIO.output(LED_BLUE, False)
        time.sleep(0.3)
        
        # 蓝
        GPIO.output(LED_RED, False)
        GPIO.output(LED_GREEN, False)
        GPIO.output(LED_BLUE, True)
        time.sleep(0.3)
    
    GPIO.output(LED_BLUE, False)
    print("\n✓ 闪烁测试完成")


def test_button():
    """测试4: 按钮测试"""
    print("\n" + "=" * 40)
    print("测试4: 按钮测试")
    print("=" * 40)
    print(f"按钮引脚: GPIO{BUTTON} (物理引脚11)")
    print("请按下按钮5次，每次按下LED会变色...")
    print("(按 Ctrl+C 跳过)\n")
    
    colors = [
        (True, False, False, "红"),
        (False, True, False, "绿"),
        (False, False, True, "蓝"),
        (True, True, False, "黄"),
        (False, False, False, "灭"),
    ]
    
    try:
        press_count = 0
        last_state = GPIO.HIGH
        
        while press_count < 5:
            current_state = GPIO.input(BUTTON)
            
            # 检测下降沿（按下）
            if last_state == GPIO.HIGH and current_state == GPIO.LOW:
                r, g, b, name = colors[press_count]
                GPIO.output(LED_RED, r)
                GPIO.output(LED_GREEN, g)
                GPIO.output(LED_BLUE, b)
                print(f"  按钮按下 #{press_count + 1} → {name}")
                press_count += 1
                time.sleep(0.3)  # 防抖
            
            last_state = current_state
            time.sleep(0.01)
        
        print("\n✓ 按钮测试完成")
        
    except KeyboardInterrupt:
        print("\n跳过按钮测试")


def test_button_continuous():
    """测试5: 持续监测按钮状态"""
    print("\n" + "=" * 40)
    print("测试5: 按钮状态监测")
    print("=" * 40)
    print("持续显示按钮状态，按 Ctrl+C 退出\n")
    
    try:
        while True:
            state = GPIO.input(BUTTON)
            status = "松开" if state == GPIO.HIGH else "按下"
            # 按下时亮绿灯
            GPIO.output(LED_GREEN, state == GPIO.LOW)
            print(f"\r按钮状态: {status}   ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        GPIO.output(LED_GREEN, False)
        print("\n\n✓ 监测结束")


def main():
    print("\n" + "=" * 50)
    print("GPIO 调试测试程序")
    print("=" * 50)
    print(f"""
引脚配置 (BCM编号):
  LED红:  GPIO{LED_RED}  (物理引脚15)
  LED绿:  GPIO{LED_GREEN} (物理引脚13)
  LED蓝:  GPIO{LED_BLUE} (物理引脚16)
  按钮:   GPIO{BUTTON} (物理引脚11)
""")
    
    setup()
    
    try:
        while True:
            print("\n" + "-" * 40)
            print("选择测试项目:")
            print("  1. LED单色测试")
            print("  2. LED组合颜色测试")
            print("  3. LED闪烁测试")
            print("  4. 按钮测试 (按5次)")
            print("  5. 按钮状态持续监测")
            print("  6. 全部测试")
            print("  0. 退出")
            print("-" * 40)
            
            choice = input("请选择 (0-6): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                test_led_individual()
            elif choice == '2':
                test_led_combined()
            elif choice == '3':
                test_led_blink()
            elif choice == '4':
                test_button()
            elif choice == '5':
                test_button_continuous()
            elif choice == '6':
                test_led_individual()
                test_led_combined()
                test_led_blink()
                test_button()
            else:
                print("无效选择")
    
    except KeyboardInterrupt:
        print("\n中断")
    
    finally:
        GPIO.output(LED_RED, False)
        GPIO.output(LED_GREEN, False)
        GPIO.output(LED_BLUE, False)
        GPIO.cleanup()
        print("\nGPIO 已清理，程序退出")


if __name__ == "__main__":
    main()
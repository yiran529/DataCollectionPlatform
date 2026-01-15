#!/usr/bin/env python3
"""
按钮诊断脚本 - 帮助确定按钮接线和配置问题
"""

import time
import sys

try:
    import Jetson.GPIO as GPIO
    PLATFORM = "Jetson"
    print("✓ 使用 Jetson.GPIO")
except ImportError:
    try:
        import RPi.GPIO as GPIO
        PLATFORM = "RaspberryPi"
        print("✓ 使用 RPi.GPIO")
    except ImportError:
        print("❌ 未找到 GPIO 库")
        sys.exit(1)

BUTTON = 17  # 物理引脚11

print("\n" + "=" * 60)
print("按钮诊断工具")
print("=" * 60)
print(f"测试引脚: GPIO{BUTTON} (物理引脚11)")
print("\n按钮接线检查清单:")
print("  ☐ 1. VCC 连接到 3.3V (物理引脚1)")
print("  ☐ 2. GND 连接到 GND (物理引脚6或14)")
print("  ☐ 3. OUT 连接到 GPIO17 (物理引脚11)")
print("=" * 60)

# 测试1: 使用上拉电阻（默认配置）
print("\n【测试1】使用内部上拉电阻 (PUD_UP)")
print("-" * 60)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("开始监测按钮状态（10秒）...")
print("请在这10秒内按下和松开按钮几次\n")

states_with_pullup = []
last_state = None
change_count = 0

for i in range(100):  # 10秒，每100ms采样
    state = GPIO.input(BUTTON)
    state_str = "HIGH" if state == GPIO.HIGH else "LOW"
    
    if state != last_state and last_state is not None:
        change_count += 1
        print(f"  [{i*100}ms] 状态变化: {state_str}")
    
    states_with_pullup.append(state)
    last_state = state
    time.sleep(0.1)

high_count_pullup = states_with_pullup.count(GPIO.HIGH)
low_count_pullup = states_with_pullup.count(GPIO.LOW)

print(f"\n结果:")
print(f"  HIGH 次数: {high_count_pullup} ({high_count_pullup}%)")
print(f"  LOW 次数:  {low_count_pullup} ({low_count_pullup}%)")
print(f"  状态变化:  {change_count} 次")

# 测试2: 使用下拉电阻
print("\n【测试2】使用内部下拉电阻 (PUD_DOWN)")
print("-" * 60)
GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

print("开始监测按钮状态（10秒）...")
print("请在这10秒内按下和松开按钮几次\n")

states_with_pulldown = []
last_state = None
change_count = 0

for i in range(100):  # 10秒
    state = GPIO.input(BUTTON)
    state_str = "HIGH" if state == GPIO.HIGH else "LOW"
    
    if state != last_state and last_state is not None:
        change_count += 1
        print(f"  [{i*100}ms] 状态变化: {state_str}")
    
    states_with_pulldown.append(state)
    last_state = state
    time.sleep(0.1)

high_count_pulldown = states_with_pulldown.count(GPIO.HIGH)
low_count_pulldown = states_with_pulldown.count(GPIO.LOW)

print(f"\n结果:")
print(f"  HIGH 次数: {high_count_pulldown} ({high_count_pulldown}%)")
print(f"  LOW 次数:  {low_count_pulldown} ({low_count_pulldown}%)")
print(f"  状态变化:  {change_count} 次")

# 分析结果
print("\n" + "=" * 60)
print("诊断结果")
print("=" * 60)

# 判断按钮类型和配置
if change_count == 0:
    print("❌ 问题: 按钮没有任何状态变化")
    print("\n可能原因:")
    print("  1. OUT引脚未连接到GPIO17（物理引脚11）")
    print("  2. 按钮模块没有供电（VCC未接3.3V）")
    print("  3. 按钮模块损坏")
    print("\n建议:")
    print("  1. 检查三条线是否都插好")
    print("  2. 使用万用表测试按钮模块:")
    print("     - 测量VCC和GND之间是否有3.3V")
    print("     - 测量OUT引脚，按下时应该变化")
    
elif high_count_pullup > 80 and low_count_pullup < 20:
    print("✓ 按钮工作正常（上拉配置）")
    print(f"\n配置: 内部上拉 (PUD_UP)")
    print(f"  - 松开时: HIGH")
    print(f"  - 按下时: LOW")
    print(f"  - 状态变化: {change_count} 次")
    
    if change_count > 0:
        print("\n✓ 按钮响应正常！")
    else:
        print("\n⚠️ 按钮未按下或未检测到变化")
        
elif high_count_pulldown > 80 and low_count_pulldown < 20:
    print("✓ 按钮工作正常（下拉配置）")
    print(f"\n配置: 内部下拉 (PUD_DOWN)")
    print(f"  - 松开时: LOW")
    print(f"  - 按下时: HIGH")
    print(f"  - 状态变化: {change_count} 次")
    
    print("\n⚠️ 需要修改代码配置:")
    print("  将 GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)")
    print("  改为 GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)")
    
else:
    print("⚠️ 状态不稳定")
    print(f"  PUD_UP:  HIGH={high_count_pullup}, LOW={low_count_pullup}")
    print(f"  PUD_DOWN: HIGH={high_count_pulldown}, LOW={low_count_pulldown}")
    print("\n可能原因:")
    print("  1. 接线松动")
    print("  2. 按钮模块故障")
    print("  3. 电压不稳定")

GPIO.cleanup()
print("\n" + "=" * 60)
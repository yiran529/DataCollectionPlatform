#!/usr/bin/env python3
"""
三引脚按钮模块诊断工具 - Jetson Xavier 版本
"""

import time
import sys

try:
    import Jetson.GPIO as GPIO
    print("✓ 使用 Jetson.GPIO")
except ImportError:
    print("❌ 未找到 Jetson.GPIO")
    sys.exit(1)

print("\n" + "=" * 70)
print("三引脚数字按钮模块诊断工具 (Jetson Xavier)")
print("=" * 70)

print("\n请确认您的按钮模块接线:")
print("-" * 70)
print("  按钮模块 VCC → Jetson 3.3V (物理引脚1)")
print("  按钮模块 GND → Jetson GND (物理引脚6 或 14)")
print("  按钮模块 OUT → Jetson GPIO引脚")
print("-" * 70)

# 让用户选择按钮连接的引脚
print("\n您的按钮 OUT 引脚连接到哪个 GPIO？")
print("  1. GPIO4  (物理引脚7)  - test_button.py 使用的引脚")
print("  2. GPIO17 (物理引脚11) - test_gpio_led.py 使用的引脚")
print("  3. 其他引脚（手动输入）")

choice = input("\n请选择 (1/2/3): ").strip()

if choice == '1':
    BUTTON = 4
    print(f"→ 测试 GPIO4 (物理引脚7)")
elif choice == '2':
    BUTTON = 17
    print(f"→ 测试 GPIO17 (物理引脚11)")
elif choice == '3':
    BUTTON = int(input("请输入 GPIO 编号 (BCM): ").strip())
    print(f"→ 测试 GPIO{BUTTON}")
else:
    print("无效选择，默认使用 GPIO17")
    BUTTON = 17

print("\n" + "=" * 70)
print(f"开始测试 GPIO{BUTTON}")
print("=" * 70)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Jetson.GPIO 不支持 pull_up_down，直接设置为输入
GPIO.setup(BUTTON, GPIO.IN)

print("\n【阶段1】初始状态检测")
print("-" * 70)
initial_state = GPIO.input(BUTTON)
print(f"当前引脚状态: {'HIGH' if initial_state == GPIO.HIGH else 'LOW'}")

if initial_state == GPIO.HIGH:
    print("\n✓ 正常: 三引脚模块通常在松开时输出 HIGH")
    print("  期望行为: 按下时应该变为 LOW")
elif initial_state == GPIO.LOW:
    print("\n⚠️ 异常: 引脚为 LOW")
    print("  可能原因:")
    print("  1. 按钮模块未供电 (VCC 未接 3.3V)")
    print("  2. 按钮一直处于按下状态")
    print("  3. OUT 引脚未连接或连接错误")

print("\n【阶段2】动态状态监测 (20秒)")
print("-" * 70)
print("请在接下来的 20 秒内，多次按下和松开按钮...")
print("(程序会实时显示状态变化)\n")

states = []
changes = []
last_state = initial_state
high_count = 0
low_count = 0

start_time = time.time()

for i in range(200):  # 20秒，每100ms采样
    state = GPIO.input(BUTTON)
    states.append(state)
    
    if state == GPIO.HIGH:
        high_count += 1
    else:
        low_count += 1
    
    # 检测状态变化
    if state != last_state:
        elapsed = (time.time() - start_time) * 1000  # 转换为毫秒
        state_str = "HIGH" if state == GPIO.HIGH else "LOW "
        changes.append((elapsed, state))
        print(f"  [{elapsed:7.1f}ms] 状态变化 → {state_str}")
        last_state = state
    
    time.sleep(0.1)

# 分析结果
print("\n" + "=" * 70)
print("测试结果")
print("=" * 70)
print(f"总采样次数:   {len(states)}")
print(f"HIGH 状态:    {high_count} 次 ({high_count*100//len(states)}%)")
print(f"LOW 状态:     {low_count} 次 ({low_count*100//len(states)}%)")
print(f"状态变化次数: {len(changes)} 次")

print("\n" + "=" * 70)
print("诊断结果与建议")
print("=" * 70)

if len(changes) == 0:
    print("\n❌ 问题: 按钮没有任何状态变化")
    
    if high_count == len(states):
        print("\n状态: 引脚始终为 HIGH")
        print("\n可能原因:")
        print("  1. ❌ 按钮 OUT 引脚未连接到 GPIO{} (物理引脚{})".format(
            BUTTON, 7 if BUTTON == 4 else (11 if BUTTON == 17 else '?')))
        print("  2. ❌ 按钮模块损坏")
        print("  3. ⚠️  您没有按下按钮")
        
        print("\n排查步骤:")
        print("  1. 检查 OUT 线是否插在正确的 GPIO 引脚上")
        print("  2. 确认 VCC 接在 3.3V (物理引脚1)")
        print("  3. 确认 GND 接在 GND (物理引脚6)")
        print("  4. 用手按住按钮，重新运行测试")
        
    elif low_count == len(states):
        print("\n状态: 引脚始终为 LOW")
        print("\n可能原因:")
        print("  1. ❌ 按钮模块未供电 (VCC 未接 3.3V)")
        print("  2. ❌ 按钮一直被按下")
        print("  3. ❌ GPIO 引脚短路到地")
        
        print("\n排查步骤:")
        print("  1. 用万用表测量 VCC 和 GND 之间是否有 3.3V")
        print("  2. 检查按钮是否卡住")
        print("  3. 断开 OUT 线，测量 OUT 引脚电压")

elif len(changes) < 4:
    print("\n⚠️ 警告: 状态变化太少")
    print(f"   检测到 {len(changes)} 次变化，建议至少 4 次")
    print("\n可能原因:")
    print("  1. 按钮接触不良")
    print("  2. 您按动按钮的次数不够")
    print("\n建议: 重新运行测试，多按几次按钮")

else:
    print("\n✅ 按钮工作正常！")
    print(f"\n检测到 {len(changes)} 次状态变化")
    
    # 分析按钮逻辑
    if high_count > low_count:
        print("\n按钮特性:")
        print("  - 默认状态 (松开): HIGH")
        print("  - 按下状态:        LOW")
        print("\n这是标准的三引脚按钮模块行为 ✓")
        
        print("\n在代码中使用:")
        print(f"  # 检测按钮按下")
        print(f"  if GPIO.input({BUTTON}) == GPIO.LOW:")
        print(f"      print('按钮被按下')")
        
    else:
        print("\n按钮特性:")
        print("  - 默认状态 (松开): LOW")
        print("  - 按下状态:        HIGH")
        print("\n这可能是反向逻辑的按钮模块")
        
        print("\n在代码中使用:")
        print(f"  # 检测按钮按下")
        print(f"  if GPIO.input({BUTTON}) == GPIO.HIGH:")
        print(f"      print('按钮被按下')")
    
    # 显示按键序列
    if changes:
        print("\n按键序列 (前10次):")
        press_count = 0
        release_count = 0
        
        for i, (ts, state) in enumerate(changes[:10]):
            if state == GPIO.LOW:
                press_count += 1
                print(f"  {i+1}. [{ts:7.1f}ms] 按下")
            else:
                release_count += 1
                print(f"  {i+1}. [{ts:7.1f}ms] 松开")
        
        if len(changes) > 10:
            print(f"  ... (还有 {len(changes)-10} 次变化)")
        
        print(f"\n统计: 按下 {press_count} 次, 松开 {release_count} 次")

# 最终建议
if len(changes) > 0:
    print("\n" + "=" * 70)
    print("✅ 下一步: 更新代码配置")
    print("=" * 70)
    print(f"\n在 test_gpio_led.py 中:")
    print(f"  1. 确认 BUTTON = {BUTTON}  # 当前测试的引脚")
    print(f"  2. 修改按钮检测逻辑:")
    
    if high_count > low_count:
        print(f"\n     # 检测下降沿（HIGH → LOW）表示按下")
        print(f"     if last_state == GPIO.HIGH and current_state == GPIO.LOW:")
        print(f"         print('按钮按下')")
    else:
        print(f"\n     # 检测上升沿（LOW → HIGH）表示按下")
        print(f"     if last_state == GPIO.LOW and current_state == GPIO.HIGH:")
        print(f"         print('按钮按下')")

GPIO.cleanup()
print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
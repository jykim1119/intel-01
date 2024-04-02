#!/usr/bin/env python3

from iotdemo import FactoryController
from pynput import keyboard

ctrl = FactoryController('/dev/ttyACM0')
led_state = False

def on_press(key):
    global led_state
    try:
        if key.char == '1':             # 키가 숫자 1인 경우 system_start()
            ctrl.system_start() 
        elif key.char == '2':           # 키가 숫자 2인 경우 system_stop()
            ctrl.system_stop()  
        elif key.char == '3':
            led_state = not led_state   # 빨간 LED 상태를 토글합니다.
            ctrl.red = led_state
        elif key.char == '4':
            led_state = not led_state    # 빨간 LED 상태를 토글합니다.
            ctrl.orange = led_state
        elif key.char == '5':
            led_state = not led_state    # 녹색 LED 상태를 토글합니다.
            ctrl.green = led_state
        elif key.char == '6':
            led_state = not led_state
            ctrl.conveyor = led_state
        elif key.char == '7':
            ctrl.push_actuator(1)
        elif key.char == '8':
            ctrl.push_actuator(2)
    except AttributeError:
        # 특수 키(숫자 1을 포함하지 않는 키)를 누를 때 예외 처리
        pass

def on_release(key):
    # 키를 눌렀다 뗄 때 실행되는 콜백
    pass

# 키보드 리스너 생성
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)

# 키보드 리스너 시작
listener.start()

# 프로그램 종료할 때 키보드 리스너 정리
listener.join()
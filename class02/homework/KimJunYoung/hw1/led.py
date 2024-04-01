"""
Smart Factory LED Test
"""

from iotdemo import FactoryController

ctrl = FactoryController('/dev/ttyACM5')

while True:
    data = input('Enter Number(1~8): ')
    if data == '1':
        ctrl.system_start()
        print("start")         
    elif data == '2':
        ctrl.system_stop()
        print("stop")
    elif data == '3':
        ctrl.red = ctrl.red
        print("Red")   
    elif data == '4':
        ctrl.orange = ctrl.orange
        print("Orange")           
    elif data == '5':
        ctrl.green = ctrl.green
        print("Green")
    elif data == '6':
        ctrl.conveyor = ctrl.conveyor
        print("Conveyor")      
    elif data == '7':
        ctrl.push_actuator(1)
        print("Actuator1")
    elif data == '8':
        ctrl.push_actuator(2)
        print("Actuator2")
    else:
        print("유효한 번호가 아닙니다.\n")
        
ctrl.close()
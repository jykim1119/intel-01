from iotdemo import FactoryController

ctrl = FactoryController('/dev/ttyACM0', debug=True)
red_state = False
orange_state = False
green_state = False
conveyor_state = False   

# 사용자로부터 명령 입력 받기
while True:
    print("어떤 작업을 수행하시겠습니까?")
    print("1. 시스템 시작")
    print("2. 시스템 끄기")
    print("3. red Test")
    print("4. orange Test")
    print("5. green Test")
    print("6. conveyor Test")
    print("7. 액추에이터 1 작동")
    print("8. 액추에이터 2 작동")
    print("9. 종료")
 
    choice = input("번호를 입력하세요: ")

    # 새로운 함수 실행
    if choice == '1':
        ctrl.system_start()
        print("system start")
    elif choice == '2':
        ctrl.system_stop()
        print("system_stop")
    elif choice == '3':
        red_state = not red_state
        ctrl.red = red_state
        print("Test :", ctrl.red)
    elif choice == '4':
        orange_state = not orange_state
        ctrl.orange = orange_state
        print("Test :", ctrl.orange)
    elif choice == '5':
        green_state  = not green_state
        ctrl.green = green_state 
        print("Test :", ctrl.green)
    elif choice == '6':
        conveyor_state = not conveyor_state
        ctrl.conveyor = conveyor_state
        print("Test :", ctrl.conveyor)
    elif choice == '7':
        ctrl.push_actuator(1)
        print("push_actuator 실행")
    elif choice == '8':
        ctrl.push_actuator(2)
        print("push_actuator 실행")
    elif choice == '9':
        print("프로그램을 종료합니다.")
        break
    else:
        print("유효하지 않은 입력입니다. 다시 시도하세요.")

# FactoryController 인스턴스 종료
ctrl.close()

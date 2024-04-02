"""
Smart Factory Test
"""
#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

from cv2 import cv2
import numpy as np
import openvino as ov

from iotdemo import ColorDetector, FactoryController, MotionDetector

FORCE_STOP = False


def thread_cam1(q):
    """
    cam1 쓰레드 동작
    """
    # MotionDetector
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg", "default")
    # Load and initialize OpenVINO
    core = ov.Core()
    model = core.read_model("./resources/openvino.xml")
    # HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("resources/conveyor.mp4")
    start_flag = False
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 Live", frame))
        # Motion detect
        detected = det.detect(frame)
        if detected is None:
            continue
        # Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(("VIDEO:Cam1 detected", detected))
        # abnormal detect
        input_tensor = np.expand_dims(detected, 0)
        if start_flag is False:
            ppp = ov.preprocess.PrePostProcessor(model)
            ppp.input().tensor() \
                .set_shape(input_tensor.shape) \
                .set_element_type(ov.Type.u8) \
                .set_layout(ov.Layout('NHWC'))
            ppp.input().preprocess() \
                .resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)
            model = ppp.build()
            compile_model = core.compile_model(model, 'CPU')
            start_flag = True
        # Inference OpenVINO
        results = compile_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)

        # in queue for moving the actuator 1
        if probs[0] < 1.5:
            print("GOOD!")
            print(f"@@{probs}")
        else:
            print("PUSH", 1)
            print(f"@@{probs}")
    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    """
    cam2 쓰레드 동작
    """
    # MotionDetector
    det = MotionDetector()
    det.load_preset("./resources/motion.cfg", "default")
    # ColorDetector
    color = ColorDetector()
    color.load_preset("./resources/color.cfg", "default")
    # HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        # HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 Live", frame))
        # Detect motion
        detected = det.detect(frame)
        if detected is None:
            continue
        # Enqueue "V IDEO:Cam2 detected", detected info.
        q.put(("VIDEO:Cam2 detected", detected))
        # Detect color
        predict = color.detect(detected)
        name, ratio = predict[0]
        ratio = ratio * 100
        # Compute ratio
        print(f"{name}: {ratio:.2f}%")
        # Enqueue to handle actuator 2
        if name == "blue" and ratio > .4:
            q.put(("PUSH", 2))
    cap.release()
    q.put(('DONE', None))
    exit()


def imshow(title, frame, pos=None):
    """
    이미지 띄우기
    """
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


def main():
    """
    main 동작 
    """
    global FORCE_STOP

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()
    # HW2 Create a Queue
    q = Queue()
    # HW2 Create thread_cam1 and thread_cam2 threads and start them.
    thread1 = threading.Thread(target=thread_cam1, args=(q,))
    thread2 = threading.Thread(target=thread_cam2, args=(q,))
    thread1.start()
    thread2.start()
    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break
            # HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            try:
                event = q.get_nowait()
            except Empty:
                continue
            # HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            name, data = event
            if name.startswith("VIDEO:"):
                imshow(name[6:], data)
            elif name == 'PUSH':
                # Control actuator, name == 'PUSH'
                ctrl.push_actuator(data)
            if name == 'DONE':
                FORCE_STOP = True

            q.task_done()
    thread1.join()
    thread2.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
        
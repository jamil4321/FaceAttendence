import numpy as np
import cv2
import os

os.environ['DISPLAY'] = ':0'
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def cam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 360), fx=0, fy=0,
                       interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()
    return (ret, frame)

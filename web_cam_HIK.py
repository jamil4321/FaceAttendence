import numpy as np
import cv2
import os

os.environ['DISPLAY'] = ':0'
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# cap = cv2.VideoCapture('rtsp://admin:Admin@123@192.168.0.64/doc/page/previw.asp',cv2.CAP_FFMPEG)
# rtsp: // admin: admin123@192.168.0.110: 554/Streaming/Channels/102/
# 'rtsp://admin:admin@192.168.0.101/mode=real&idc=1&ids=1'


def cam():
    cap = cv2.VideoCapture('http://192.168.0.209/capture', cv2.CAP_FFMPEG)
    # cap = cv2.VideoCapture(
    #     'rtsp://admin:L2658DCE@192.168.0.210:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif')
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 360), fx=0, fy=0,
                       interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()
    return (ret, frame)


# def cam_new():
#     while True:
#         # cap = cv2.VideoCapture('http://192.168.0.3/capture', cv2.CAP_FFMPEG)
#         cap = cv2.VideoCapture(0)
#         ret, frame = cap.read()
#         frame = cv2.resize(frame, (640, 360), fx=0, fy=0,
#                            interpolation=cv2.INTER_CUBIC)
#         # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         cv2.imshow('frame', frame)
#     cap.release()


# cam_new()

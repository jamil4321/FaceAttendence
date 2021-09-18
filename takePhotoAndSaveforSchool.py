import cv2
import os
import paho.mqtt.client as paho
import requests
import json
from send2trash import send2trash


os.environ['DISPLAY'] = ':0'


broker = "192.168.0.205"
port = 1883


def on_publish(client, userdata, result):  # create function for callback
    print("data published \n")
    pass


def on_message(client, userdata, msg):
    print(msg.payload.decode())
    print(msg.topic)

    if msg.topic == 'S1001/Places/Indoor/School/NeMember':
        face_ascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        flag = True
        while flag:
            cap = cv2.VideoCapture(
                'http://192.168.0.209/capture', cv2.CAP_FFMPEG)
            _, frame = cap.read()

            faces = face_ascade.detectMultiScale(frame, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                print('faces Found'+str(len(faces))+'')
                if not os.path.exists('./ids/'+msg.payload.decode()):
                    os.makedirs('./ids/'+msg.payload.decode())
                cv2.imwrite('./ids/'+msg.payload.decode() +
                            '/'+'face'+'.jpg', frame[y: y + h, x: x + w])
                flag = False
                print(flag)
            cv2.imshow('frame', frame)
            cv2.waitKey(30)
        cap.release()
        cv2.destroyAllWindows()
    elif msg.topic == 'S1001/Places/Indoor/School/DeleteMember':
        send2trash('./ids/'+msg.payload.decode())
    else:
        print('Sorry')


client1 = paho.Client("AddadnRemoveCamera")
client1.connect(broker, port)
client1.on_publish = on_publish
client1.subscribe("S1001/Places/Indoor/School/NeMember")
client1.subscribe("S1001/Places/Indoor/School/DeleteMember")
client1.on_message = on_message
client1.loop_forever()

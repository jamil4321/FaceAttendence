from requests.api import get
from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.python.platform import gfile
import numpy as np
import detect_and_align
import argparse
import cv2
import os
import tensorflow.compat.v1 as tf
import urllib
import os
import cv2
import web_cam_HIK
import paho.mqtt.client as paho
import time
from threading import Thread
import requests
import json
import asyncio
import socketio

sio = socketio.Client()
sio.connect('http://localhost:2000')

tf.disable_v2_behavior()
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['DISPLAY'] = ':0'
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
broker = "192.168.0.205"
port = 1883
token = ''
# client1 = ''
# getToken = requests.get('https://gym.iotsol.pk/token')
# getToken = getToken.json()
# my_headers = {'Authorization': 'Bearer '+getToken['token']}
# print(my_headers)


def on_message(client, userdata, msg):
    print(msg.payload.decode())
    print(msg.topic)
    if msg.topic == 'G1001/Places/Indoor/Gym/Train':
        if msg.payload.decode() == 'Train':
            # with tf.Graph().as_default():
            with tf.Session() as sess:
                sess.close()
                main(parser.parse_args())


class IdData:
    """Keeps track of known identities and calculates id matches"""

    def __init__(
        self, id_folder, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, distance_treshold
    ):
        print("Loading known identities: ")
        self.distance_treshold = distance_treshold
        self.id_folder = id_folder
        self.mtcnn = mtcnn
        self.id_names = []
        self.embeddings = None

        image_paths = []
        os.makedirs(id_folder, exist_ok=True)
        ids = os.listdir(os.path.expanduser(id_folder))
        if not ids:
            return

        for id_name in ids:
            id_dir = os.path.join(id_folder, id_name)
            image_paths = image_paths + \
                [os.path.join(id_dir, img) for img in os.listdir(id_dir)]

        print("Found "+str(len(image_paths))+" images in id folder")
        aligned_images, id_image_paths = self.detect_id_faces(image_paths)
        feed_dict = {images_placeholder: aligned_images,
                     phase_train_placeholder: False}
        self.embeddings = sess.run(embeddings, feed_dict=feed_dict)

        # if len(id_image_paths) < 5:
        self.print_distance_table(id_image_paths)

    def add_id(self, embedding, new_id, face_patch):
        if self.embeddings is None:
            self.embeddings = np.atleast_2d(embedding)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        self.id_names.append(new_id)
        id_folder = os.path.join(self.id_folder, new_id)
        os.makedirs(id_folder, exist_ok=True)
        filenames = [s.split(".")[0] for s in os.listdir(id_folder)]
        numbered_filenames = [int(f) for f in filenames if f.isdigit()]
        img_number = max(numbered_filenames) + 1 if numbered_filenames else 0
        cv2.imwrite(os.path.join(id_folder, face_patch+".jpg"))

    def detect_id_faces(self, image_paths):
        aligned_images = []
        id_image_paths = []
        for image_path in image_paths:
            image = cv2.imread(os.path.expanduser(
                image_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_patches, _, _ = detect_and_align.detect_faces(
                image, self.mtcnn)
            if len(face_patches) > 1:
                print(
                    "Warning: Found multiple faces in id image: %s" % image_path
                    + "\nMake sure to only have one face in the id images. "
                    + "If that's the case then it's a false positive detection and"
                    + " you can solve it by increasing the thresolds of the cascade network"
                )
            aligned_images = aligned_images + face_patches
            id_image_paths += [image_path] * len(face_patches)
            path = os.path.dirname(image_path)
            self.id_names += [os.path.basename(path)] * len(face_patches)

        return np.stack(aligned_images), id_image_paths

    def print_distance_table(self, id_image_paths):
        """Prints distances between id embeddings"""
        distance_matrix = pairwise_distances(self.embeddings, self.embeddings)
        image_names = [path.split("/")[-1] for path in id_image_paths]
        print("Distance matrix:\n{:20}".format(""), end="")
        [print("{:20}".format(name), end="") for name in image_names]
        for path, distance_row in zip(image_names, distance_matrix):
            print("\n{:20}".format(path), end="")
            for distance in distance_row:
                print("{:20}".format("%0.3f" % distance), end="")
        print()

    def find_matching_ids(self, embs):
        if self.id_names:
            matching_ids = []
            matching_distances = []
            distance_matrix = pairwise_distances(embs, self.embeddings)
            for distance_row in distance_matrix:
                min_index = np.argmin(distance_row)
                if distance_row[min_index] < self.distance_treshold:
                    matching_ids.append(self.id_names[min_index])
                    matching_distances.append(distance_row[min_index])
                else:
                    matching_ids.append(None)
                    matching_distances.append(None)
        else:
            matching_ids = [None] * len(embs)
            matching_distances = [np.inf] * len(embs)
        return matching_ids, matching_distances


def load_model(model):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print("Loading model filename: %s" % model_exp)
        with gfile.FastGFile(model_exp, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    else:
        raise ValueError("Specify model file, not directory!")


async def threadedFunc(frame, mtcnn, images_placeholder, phase_train_placeholder, sess, embeddings, id_data, client1):
    # with tf.Graph().as_default():
    with tf.Session() as sess:
        face_patches, padded_bounding_boxes, landmarks = detect_and_align.detect_faces(
            frame, mtcnn)
        if len(face_patches) > 0:
            face_patches = np.stack(face_patches)
            feed_dict = {images_placeholder: face_patches,
                         phase_train_placeholder: False}
            embs = sess.run(embeddings, feed_dict=feed_dict)
            matching_ids, matching_distances = id_data.find_matching_ids(
                embs)
            print("Matches in frame:")
            for bb, matching_id, dist in zip(padded_bounding_boxes, matching_ids, matching_distances):
                if matching_id is None:
                    matching_id = "Unknown"
                    print("Unknown! Couldn't find match.")
                else:
                    data = {
                        "studentID": matching_id,
                        "cameraID": "1"
                    }
                    send_id = requests.post(
                        'http://127.0.0.1:2000/api/markAttendence', json=data)
                    sio.emit('Mark Attendence')
                    # send_id = send_id.json()
                    print(send_id.json())
                    print("Hi %s! Distance: %1.4f" % (matching_id, dist))
        else:
            print("Couldn't find a face")


async def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:

            # Setup models
            mtcnn = detect_and_align.create_mtcnn(sess, None)
            model = './folder/model.pb'
            id_folder = './ids/'
            load_model(model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Load anchor IDs
            id_data = IdData(
                id_folder, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, args.threshold
            )
            faceCascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            client1 = paho.Client("ModelTrainClient")
            client1.connect(broker, port)
            client1.subscribe("G1001/Places/Indoor/Gym/Train")
            client1.on_message = on_message
            client1.loop_start()
            while True:
                ret, frame = web_cam_HIK.cam()

                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    for (x, y, w, h) in faces:
                        cv2.rectangle(
                            frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        await threadedFunc(frame, mtcnn, images_placeholder,
                                           phase_train_placeholder, sess, embeddings, id_data, client1)
                    cv2.imshow("frame", frame)
                    key = cv2.waitKey(1)
                    cv2.waitKey(1)
                    if key == ord("q"):
                        break
                else:
                    print('Camera not found')
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", type=float,
                        help="Distance threshold defining an id match", default=1.0)
    asyncio.run(main(parser.parse_args()))
    # while True:
    #     try:
    #         main(parser.parse_args())
    #     except:
    #         main(parser.parse_args())

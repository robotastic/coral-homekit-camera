import zmq
import threading


from collections import defaultdict
from PIL import Image
import json
from edgetpu.learn.imprinting.engine import ImprintingEngine
import numpy as np
from tinydb import TinyDB
import uuid
import picamera
import io
import os

def get_labels():
    if os.path.exists("./models/map.json"):
        with open("./models/map.json") as f:
            try:    
                labels = json.loads(f.read())
                #labels = dict((v,k) for k,v in labels.items())
                return labels
            except:
                return False
    else:
        return False

def retrain(model='./models/mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite', out_file='./models/classify.tflite' , map_file='./models/map.json'):
    train_dict = defaultdict(lambda: [])
    train_input = []
    labels_map = {}
    train_set = defaultdict(lambda: [])
    pics = TinyDB("./pics.json")
    for pic in pics:
        train_set[pic["class"]].append(pic["img"])
        print(pic)
    samples = pics.all()
    for class_id, (set) in enumerate(train_set):
        print('Processing Class: ', set, class_id)
        ret = []
        for filename in train_set[set]:
            img=Image.open("./pics/{}.jpg".format(filename)).resize((224,224))
            ret.append(
                np.asarray(img).flatten()
            )
        train_input.append(np.array(ret))
        labels_map[class_id] = set

    if (len(samples) == 0) or not (("background" in train_set.keys()) and ("detection" in train_set.keys()) ):
        print("Training data not good")
        return False

    else:
        print("Enough data, going to try and retrain")
        engine = ImprintingEngine(model)
        print("Engines ready")
        engine.TrainAll(train_input)
        print("ReTraining complete")
        with open("./models/map.json", 'w') as outfile:
            json.dump(labels_map, outfile)
        print("Saved Labels")
        engine.SaveModel(out_file)
        print("Model Saved")
        return True


class StateManager(object):
    def __init__(self):
        self.context = zmq.Context()
        self.zmq_socket = self.context.socket(zmq.PUSH)
        self.zmq_socket.bind("tcp://127.0.0.1:5557")
        self.last_state = "run"

    def collect_background(self):
        work_message = {"state": "collect_background"}
        self.last_state = "collect"
        self.zmq_socket.send_json(work_message)

    def collect_detection(self):
        work_message = {"state": "collect_detection"}
        self.last_state = "collect"
        self.zmq_socket.send_json(work_message)

    def retrain(self):
        work_message = {"state": "retrain"}
        self.last_state = "retrain"
        self.zmq_socket.send_json(work_message)

    def shutdown(self):
        work_message = {"state": "shutdown"}
        self.last_state = "shutdown"
        self.zmq_socket.send_json(work_message)

    def run(self):
        work_message = {"state": "run"}
        self.last_state = "run"

        self.zmq_socket.send_json(work_message)


class ApplicationState(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.context = zmq.Context()
        self.consumer_receiver = self.context.socket(zmq.PULL)
        self.consumer_receiver.connect("tcp://127.0.0.1:5557")
        self.last_state = "run"
        self.is_running = True
    def run(self):
        while self.is_running:
            message = self.consumer_receiver.recv_json()
            self.last_state = message["state"]
            if self.last_state == "shutdown":
                self.is_running = False


class Camera(object):

    def __init__(self):
        self.cam = picamera.PiCamera()
        self.cam.resolution = (640, 480)
        self.cam.vflip = True
        self.pics = TinyDB("./pics.json")

    def collect(self, pclass):
        uid = str(uuid.uuid4())
        self.cam.capture("./pics/{}.jpg".format(uid))
        self.pics.insert({"class": pclass, "img": uid})

    def returnPIL(self):
        stream = io.BytesIO()
        self.cam.capture(stream, format='jpeg')
        stream.seek(0)
        img = Image.open(stream).resize((224, 224))
        return img

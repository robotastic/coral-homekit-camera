import logging

from edgetpu.classification.engine import ClassificationEngine
import io
from utils import retrain, get_labels,Camera, ApplicationState
import signal
import os

camera = Camera()
stream = io.BytesIO()
app_state = ApplicationState()
app_state.start()


class MotionSensor():


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_trained = retrain()
        if self.is_trained:
            print("Model is trained and ready to go")
            self.engine = ClassificationEngine("./models/classify.tflite")
        else:
            print("Still need a custom model")
            self.engine = False #ClassificationEngine("./models/classify.tflite")
        self.labels = get_labels()
        self.is_running = True

    def run(self):
        while self.is_running:

            #print("state: ", app_state.last_state)
            if app_state.last_state == "shutdown":
                self.is_running = False
                os.system('kill $PPID')

            if (app_state.last_state == "run") and self.is_trained:
                detection=False
                img = camera.returnPIL()
                output = self.engine.ClassifyWithImage(img)
                for detect in output:
                    print("Detection: ", detect)
               # if output[0][0] == int(self.labels["detection"]):
               #     detection = True
               #     logging.info("detection triggered")
                self._detected(detection)

            if app_state.last_state == "retrain":
                logging.info("imprinting weights")
                self.is_trained=retrain()
                self.labels = get_labels()

                if self.is_trained:
                    self.engine = ClassificationEngine("./models/classify.tflite")
                    app_state.last_state = "run"
                    logging.info("finished imprinting")

                else:
                    app_state.last_state = "collect"
                    logging.warning("could not imprint weights. Please provide enough pictures")

            if app_state.last_state == "collect_background":
                camera.collect("background")
                app_state.last_state = "collect"

            if app_state.last_state == "collect_detection":
                camera.collect("detection")
                app_state.last_state = "collect"

    def _detected(self, val=False):
        if val:
            print("Detection")
        #self.char_detected.set_value(val)

    def stop(self):
        logging.info("shut down")
        super().stop()


logging.basicConfig(level=logging.INFO, format="[%(module)s] %(message)s")




# We want SIGTERM (terminate) to be handled by the driver itself,
# so that it can gracefully stop the accessory, server and advertising.

# Start it!
sensor = MotionSensor()
sensor.run()

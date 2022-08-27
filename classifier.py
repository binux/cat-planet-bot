import json
import time
import cv2 as cv
import numpy as np
import tensorflow as tf


class CatCityClassifier:

  def __init__(self):
    self.load_model()
    self.last_predict_time = None
    self.last_predict_result = None

  def load_model(self):
    interpreter = tf.lite.Interpreter(model_path='classifier.tflite')
    self.classifier_lite = interpreter.get_signature_runner('serving_default')
    with open('class_names.json') as fp:
      self.classifier_names = json.load(fp)

  def predict(self, frame):
    input_details = self.classifier_lite.get_input_details()
    _, h, w, _ = input_details['input_input']['shape']
    frame = cv.resize(frame, (w, h))
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)
    rgb_tensor = tf.expand_dims(rgb_tensor, 0)

    predictions = self.classifier_lite(input_input=rgb_tensor)['outputs']
    score = tf.nn.softmax(predictions[0])
    return self.classifier_names[np.argmax(score)], 100 * np.max(score)

  def lazy_predict(self, frame):
    if self.last_predict_time and time.time() - self.last_predict_time < 0.5:
      return self.last_predict_result
    ret = self.predict(frame)
    self.last_predict_time = time.time()
    self.last_predict_result = ret
    return ret


if __name__ == '__main__':
  classifier = CatCityClassifier()

  import glob
  import sys

  for file in glob.glob(sys.argv[1] if len(sys.argv) > 1 else
                        'data/labeled/battle_result_failed/*.png'):
    frame = cv.imread(file)
    ret = classifier.predict(frame)
    print(ret)

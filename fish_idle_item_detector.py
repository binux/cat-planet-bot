import json
import pytesseract
import re
import numpy as np
import cv2 as cv

class FishIdleItemDetector:
  def __init__(self, debug=False):
    self.debug = debug
    data = json.load(open('fish_idle_item_nums.json'))
    self.region_boxes = data['region_boxes']

  def detect(self, frame):
    result = []
    crops = [frame[y:y + h, x:x + w] for x, y, w, h in self.region_boxes]
    for img in crops:
      img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      _, img = cv.threshold(img, 170, 255, cv.THRESH_BINARY)
      text = pytesseract.image_to_string(
          img,
          lang='snum',
          config="--psm 7 -c tessedit_char_whitelist=x0123456789")
      match = re.search(r'\d+', text)
      if match:
        result.append(int(match.group(0)))
      else:
        result.append(None)
    if self.debug:
      for x, y, w, h in self.region_boxes:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return result

if __name__ == '__main__':
  detector = FishIdleItemDetector(debug=True)

  import glob
  for file in glob.glob('data/labeled/fish_idle/*.png'):
    frame = cv.imread(file)
    result = detector.detect(frame)
    print(file, result)
    cv.imshow('frame', frame)
    cv.waitKey()

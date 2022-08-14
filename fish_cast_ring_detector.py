import cv2 as cv
import numpy as np

class FishCastRingDetector:
  def __init__(self, debug=False):
    self.debug = debug

    circle = np.array([np.cos(np.arange(0, 2*np.pi, 2*np.pi/100)), np.sin(np.arange(0, 2*np.pi, 2*np.pi/100))])
    self.circle = circle.swapaxes(0, 1)

    self.target_radius = 285

  def detect(self, frame):
    img = cv.inRange(frame, (190, 190, 190), (255, 255, 255))
    img = img[220:1000,400:1100]

    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
      return
    contours = sorted(contours,
                      key=lambda c: cv.arcLength(c, True),
                      reverse=True)[:5]
    best = min(
        contours,
        key=lambda c: cv.matchShapes(self.circle, c, cv.CONTOURS_MATCH_I1, 0))
    if len(best) < 10:
      return
    ellipse = cv.fitEllipse(best)
    center, size, angle = ellipse
    center_distance = cv.norm(np.array(center) - (372, 366))
    if center_distance < 30 and abs(size[0] - size[1]) < 30:
      if self.debug:
        cv.ellipse(frame, (np.array(center) + (400, 220), size, angle),
                  (255, 0, 0), 2)
      return min(size[0], size[1])

if __name__ == '__main__':
  detector = FishCastRingDetector(True)

  import glob
  for file in glob.glob('data/labeled/fish_ring/*.png'):
    frame = cv.imread(file)
    radius = detector.detect(frame)
    print(file, radius)
    cv.imshow('frame', frame)
    cv.waitKey()

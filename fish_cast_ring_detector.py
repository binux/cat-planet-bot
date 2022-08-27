import cv2 as cv
import numpy as np

class FishCastRingDetector:
  def __init__(self, debug=False):
    self.debug = debug


    circle = np.array([np.cos(np.arange(0, 2*np.pi, 2*np.pi/100)), np.sin(np.arange(0, 2*np.pi, 2*np.pi/100))])
    self.circle = circle.swapaxes(0, 1)
    self.region = [[400, 220], [1100, 1000]]
    self.logo_mask = np.s_[290:430,:]
    self.center = (373.1545715332031, 366.3962707519531)

  def detect(self, frame):
    region = self.region
    filter_color = np.array((176, 158, 83))

    img = frame[region[0][1]:region[1][1], region[0][0]:region[1][0]]
    img = cv.inRange(img, filter_color - 30, filter_color + 30)
    kernel = np.ones((5,5),np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    # img = cv.erode(img, kernel, iterations=1)
    target_radius_img = img
    # cv.imshow('target_radius_img', target_radius_img)

    img = frame[region[0][1]:region[1][1], region[0][0]:region[1][0]]
    img = cv.inRange(img, (190, 190, 190), (255, 255, 255))
    img[self.logo_mask] = 0
    radius_img = img
    # cv.imshow('radius_img', radius_img)

    # target_radius = self.fitEllipse(frame, target_radius_img)
    # radius = self.fitEllipse(frame, radius_img)
    target_radius = self.findRadius(frame, target_radius_img)
    radius = self.findRadius(frame, radius_img)
    return target_radius and target_radius - 10, radius

  def findRadius(self, frame, img):
    img = cv.Canny(img, 50, 200, None, 3)
    distance = np.linalg.norm(np.transpose(np.nonzero(img > 0)) - self.center,
                              axis=1)
    hist, edge = np.histogram(distance, 10, (60, 300))
    # print(hist, edge)
    if np.max(hist) < 100:
      return
    radius_l, radius_h = edge[np.argmax(hist)], edge[np.argmax(hist) + 1]
    # print(radius_l, radius_h)
    radius = np.mean(distance[(radius_l < distance) & (distance < radius_h)])
    # print(radius)
    radius = np.mean(distance[(radius - 10 < distance)
                              & (distance < radius + 50)])
    if self.debug:
      cv.circle(frame, np.intp(np.array(self.center) + self.region[0]),
                int(radius), (0, 255, 0), 2)
    return radius * 2

  def fitEllipse(self, frame, img):
    circle = self.circle

    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
      return
    contours = sorted(contours,
                      key=lambda c: cv.arcLength(c, True),
                      reverse=True)[:5]
    contours = sorted(
        contours, key=lambda c: cv.matchShapes(circle, c, cv.CONTOURS_MATCH_I1, 0))
    good_contours = []
    for cont in contours:
      if len(cont) < 10:
        continue
      ellipse = cv.fitEllipse(cont)
      center, size, angle = ellipse
      if abs(center[0] - self.center[0]) and min(size) > 50:
        good_contours.append(cont)
    if not good_contours:
      return
    ellipse = cv.fitEllipse(np.vstack(good_contours))
    center, size, angle = ellipse
    if self.debug:
      cv.ellipse(frame, (np.array(center) + (400, 220), size, angle),
                (255, 0, 0), 2)
    return np.mean(size)

if __name__ == '__main__':
  import sys
  detector = FishCastRingDetector(True)

  import glob
  for file in glob.glob(
      sys.argv[1] if len(sys.argv) > 1 else 'data/labeled/fish_ring/*.png'):
    frame = cv.imread(file)
    radius = detector.detect(frame)
    print(file, radius)
    cv.imshow('frame', frame)
    if cv.waitKey() & 0xff == ord('q'):
      break

import json
import cv2 as cv
import numpy as np

class AnchorDetector:

  def __init__(self, debug=False):
    self.debug = debug

    model = json.load(open('anchor_detector.json'))
    self.bounding_rect = np.array(model['bounding_rect'])
    self.gauge_center = np.array(model['gauge_center'])
    self.gauge_radius = model['gauge_radius']
    self.anchor_mask = np.array(model['anchor_mask'], dtype=np.uint8)
    self.left_deg = model['left_deg']
    self.right_deg = model['right_deg']
    # self.target_deg = [68.66, 75.79]
    self.target_deg = 71

    start, end = sorted([self.left_deg, self.right_deg])
    step = (end - start) / 100
    self.masks = []
    for angle in np.arange(start, end, step):
      ret = self.rotated_anchor_mask(angle)
      if ret is None:
        continue
      mask, box = ret
      self.masks.append((angle, mask, box))

  def rotated_anchor_mask(self, angle):
    contour_box = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    # create mask to bounding_rect
    x, y, w, h = self.bounding_rect
    mask = np.zeros((h, w), np.uint8)
    mask_center = np.array(self.gauge_center) - np.array((x, y))
    # rotate anchor
    anchor_radius = int(self.anchor_mask.shape[0]/2)
    rot_mat = cv.getRotationMatrix2D((anchor_radius, anchor_radius), angle - 90, 1)
    rotated_anchor_mask = cv.warpAffine(self.anchor_mask.copy(), rot_mat,
                                        [anchor_radius * 2, anchor_radius * 2])
    # place anchor
    rad = angle / 180 * np.pi
    anchor_center_radius = self.gauge_radius + 26  # = (72 - 5 blur - 15 tail) / 2
    vector = np.array((np.cos(rad), -np.sin(rad))) * anchor_center_radius
    anchor_center = np.int32(mask_center + vector)
    anchor_top_left = anchor_center - anchor_radius
    anchor_box = np.hstack(
        [anchor_top_left + (x, y), (anchor_radius * 2, anchor_radius * 2)])
    # calc intersect
    ret, intersect = cv.intersectConvexConvex(
        np.int32(np.array([0, 0]) + contour_box * [w, h]),
        np.int32(anchor_top_left + contour_box * anchor_radius * 2))
    if not ret:
      return None
    intersect = intersect.reshape(4, 2)

    top_left = np.int32(intersect[np.argmin(intersect.sum(1))])
    bottom_right = np.int32(intersect[np.argmax(intersect.sum(1))])
    # print(top_left, bottom_right, mask[top_left[1]:bottom_right[1],
    #                                    top_left[0]:bottom_right[0]].shape)
    indices = np.indices(rotated_anchor_mask.shape).swapaxes(0, 2).swapaxes(0, 1)
    indices = np.logical_and(
        np.all((top_left - anchor_top_left)[::-1] <= indices, 2),
        np.all(indices < (bottom_right - anchor_top_left)[::-1], 2))
    cv.copyTo(
        rotated_anchor_mask[indices].reshape((bottom_right - top_left)[::-1]),
        None, mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])
    return mask, anchor_box

  def detect(self, frame):
    x, y, w, h = self.bounding_rect
    test_image = frame[y:y + h, x:x + w]
    test_image = cv.inRange(test_image, (240, 240, 240), (255, 255, 255))

    results = []
    for angle, mask, box in self.masks:
      score = cv.countNonZero(cv.bitwise_and(test_image, mask))
      results.append((score, angle, mask, box))
    if not results:
      return None
    score, angle, mask, box = max(results, key=lambda x: x[0])
    if score <= 0:
      return None
    if self.debug:
      cv.rectangle(frame, box[:2], box[:2] + box[2:], (255, 0, 0), 2)
    return angle

if __name__ == '__main__':
  detector = AnchorDetector(debug=True)

  import glob
  for file in glob.glob('data/labeled/fish_drag/*.png'):
    frame = cv.imread(file)
    angle = detector.detect(frame)
    print(file, angle)
    cv.imshow('frame', frame)
    cv.waitKey()
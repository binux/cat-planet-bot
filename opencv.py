import time
import cv2
import numpy as np

class VideoCapture:
  debug_color = (0, 255, 0)

  def __init__(self):
    self.frame = None
    self.prev_frame_time = None
    self.last_screen_corners = None

  @staticmethod
  def get_intersect(a1, a2, b1, b2):
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
      return None
    return (x/z, y/z)

  def screen_detection(self):
    assert self.frame is not None
    fx, fy = self.frame.shape[1] / 320, self.frame.shape[0] / 180
    f = cv2.resize(self.frame.copy(), (320, 180))
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    f = cv2.GaussianBlur(f, (5, 5), 0)
    cv2.imshow('f', f)
    # _, f = cv2.threshold(f, 60, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(f, 50, 200, None, 3)
    cv2.imshow('edged', edged)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_TC89_L1)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
      if cv2.contourArea(c) < 320*180 / 4:
        continue
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.05 * peri, True)
      if len(approx) == 4:
        ret = approx.reshape(4, 2) * np.int32((fx, fy))
        cv2.drawContours(self.frame, [ret], 0, self.debug_color, 5)
        return ret
    else:
      return None

  def keystone(self, corners, size=(1280, 720)):
    if corners is None:
      return self.frame
    dst = np.float32([[0, size[1]], size, [size[0], 0], [0, 0]])
    matrix = cv2.getPerspectiveTransform(np.float32(corners), dst)
    self.frame = cv2.warpPerspective(self.frame, matrix, size)
    return self.frame

  def add_fps(self):
    # Add FPS info
    now = time.time()
    if self.prev_frame_time:
      fps = int(1 // (now - self.prev_frame_time))
      cv2.putText(self.frame, str(fps), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3,
                  self.debug_color)
    self.prev_frame_time = now

  def run(self):
    vid = cv2.VideoCapture(0)

    while(True):
      # Capture the video frame
      # by frame
      ret, frame = vid.read()
      if not ret:
        time.sleep(100)
        continue

      frame = self.process_frame(frame)

      # Display the resulting frame
      cv2.imshow('frame', frame)

      keycode = cv2.waitKey(1) & 0xFF
      if keycode == ord('q'):
        break
      if keycode == ord('s'):
        cv2.imwrite('image.bmp', frame)

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

  def process_frame(self, frame):
    self.frame = frame
    screen_corners = self.screen_detection()
    if screen_corners is not None:
      self.last_screen_corners = screen_corners
    self.keystone(self.last_screen_corners)
    self.add_fps()
    return self.frame

  def debug(self, image_filename):
    frame = cv2.imread(image_filename)
    self.process_frame(frame)
    cv2.imwrite('image_processed.bmp', self.frame)
    cv2.imshow('frame', self.frame)
    cv2.waitKey(0)


if __name__ == '__main__':
  VideoCapture().run()
  # VideoCapture().debug('image.bmp')

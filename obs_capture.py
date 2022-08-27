import cv2 as cv

class OBSCapture:
  def __init__(self):
    self.cap = cv.VideoCapture(0)
    if not self.cap.isOpened():
      raise Exception("Cannot open camera")
    self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1546)
    self.cap.set(cv.CAP_PROP_FRAME_HEIGHT,1080)
    self.cap.set(cv.CAP_PROP_BUFFERSIZE, 2)

  def reset_windows_pos(self):
    pass

  def capture(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      frame = cv.copyMakeBorder(frame,
                                60,
                                0,
                                0,
                                0,
                                cv.BORDER_CONSTANT,
                                value=(255, 255, 255))
      yield frame

if __name__ == '__main__':
  import time
  import datetime

  screen_capture = OBSCapture()

  last_time = 0
  for frame in screen_capture.capture():
    # print('fps = %.1f' % (1 / (time.time() - last_time)))
    # last_time = time.time()

    cv.imshow("view", frame)
    key_code = cv.waitKey(1) & 0xFF
    if key_code == ord("q"):
      cv.destroyAllWindows()
      break
    elif key_code == ord("s"):
      cv.imwrite(
          'data/raw/%s.png' %
          datetime.datetime.now().isoformat().replace(':', '.'), frame)

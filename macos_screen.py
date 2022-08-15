import applescript
import collections
import datetime
import mss
import time
import cv2 as cv
import numpy as np



class ScreenCapture:
  debug_color = (0, 255, 0)
  frame_size = 1140 * 1546
  title_hight = 60


  def __init__(self, window_name='LetsView'):
    self.window_name = window_name
    self._window_pos = None

  @property
  def windows_pos(self):
    if self._window_pos is None:
      self._window_pos = self.get_window_pos()
    return self._window_pos

  def reset_windows_pos(self):
    self._window_pos = None

  def get_window_pos(self):
    return applescript.AppleScript('''
    if application "{0}" is running then
      tell application "System Events" to tell application process "{0}"
        get {{position, size}} of front window
      end tell
    end if
    '''.format(self.window_name)).run()

  def capture(self):
    with mss.mss() as sct:
      while True:
        if not self.windows_pos:
          time.sleep(1)
        (l, t), (w, h) = self._window_pos
        monitor = {"top": t, "left": l, "width": w, "height": h}
        frame = np.array(sct.grab(monitor))
        frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
        yield frame


if __name__ == '__main__':
  save = False
  last_save = 0
  screen_capture = ScreenCapture('LetsView')

  last_time = 0
  for frame in screen_capture.capture():
    print('fps = %.1f' % (1 / (time.time() - last_time)))
    last_time = time.time()
    if save and last_save < time.time() - 1:
      cv.imwrite('data/%s.png' % datetime.datetime.now().isoformat(), frame)
      last_save = time.time()

    cv.imshow("view", frame)
    key_code = cv.waitKey(1) & 0xFF
    if key_code == ord("q"):
      cv.destroyAllWindows()
      break
    elif key_code == ord("s"):
      save = not save
    elif key_code == ord("r"):
      screen_capture.reset_windows_pos()

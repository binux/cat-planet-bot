import collections
import datetime
import mss
import time
import cv2 as cv
import numpy as np
import os
import sys



class ScreenCapture:
  debug_color = (0, 255, 0)
  frame_dim = (1546, 1140)
  title_hight = 60

  def __init__(self, window_name=None):
    self.window_name = window_name
    self._window_pos = None

  @property
  def windows_pos(self):
    if self._window_pos is None:
      ret = self.get_window_pos()
      if not ret:
        return
      (l, t), (w, h) = ret
      self._window_pos = {"top": t, "left": l, "width": w, "height": h}
    return self._window_pos

  def reset_windows_pos(self):
    self._window_pos = None

  def get_window_pos(self):
    """position, size of client area"""
    if sys.platform == 'win32':
      return self._win_get_window_pos()
    elif sys.platform == 'darwin':
      return self._mac_get_window_pos()

  def _mac_get_window_pos(self):
    import applescript
    window_name = self.window_name or 'LetsView'
    ret = applescript.AppleScript('''
    if application "{0}" is running then
      tell application "System Events" to tell application process "{0}"
        get {{position, size}} of front window
      end tell
    end if
    '''.format(window_name)).run()
    if ret:
      (l, t), (w, h) = self._window_pos
      return [(l, t + 60), (w, h - 60)]

  def _win_get_window_pos(self):
    import ctypes
    window_name = self.window_name or 'AirServer® Universal (x64)'
    hwnd = ctypes.windll.user32.FindWindowW(0, window_name)
    if not hwnd:
      return None
    client_rect = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.pointer(client_rect))
    # print(client_rect.left, client_rect.top, client_rect.right, client_rect.bottom)
    # window_rect = ctypes.wintypes.RECT()
    # ctypes.windll.user32.GetWindowRect(hwnd, ctypes.pointer(window_rect))
    # print(window_rect.left, window_rect.top, window_rect.right, window_rect.bottom)
    frame_rect = ctypes.wintypes.RECT()
    DWMWA_EXTENDED_FRAME_BOUNDS = 9
    ctypes.windll.dwmapi.DwmGetWindowAttribute(
        hwnd, ctypes.wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
        ctypes.byref(frame_rect), ctypes.sizeof(frame_rect))
    # print(frame_rect.left, frame_rect.top, frame_rect.right, frame_rect.bottom)
    # I have no idea
    return [
        (frame_rect.left + client_rect.left + 2,
         frame_rect.bottom - (client_rect.bottom - client_rect.top) - 1),
        (client_rect.right - client_rect.left,
         client_rect.bottom - client_rect.top),
    ]

  def capture(self):
    with mss.mss() as sct:
      while True:
        if not self.windows_pos:
          time.sleep(1)
          continue
        frame = np.array(sct.grab(self.windows_pos))
        if not frame.size:
          continue
        frame = cv.resize(frame, (1546, 1080))
        frame = cv.copyMakeBorder(frame,
                                  60,
                                  0,
                                  0,
                                  0,
                                  cv.BORDER_CONSTANT,
                                  value=(255, 255, 255))
        frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)
        yield frame


if __name__ == '__main__':
  save = False
  last_save = 0
  # screen_capture = ScreenCapture('AirServer Windows 10 Desktop Edition')
  screen_capture = ScreenCapture()

  last_time = 0
  for frame in screen_capture.capture():
    print(frame.shape)
    # print('fps = %.1f' % (1 / (time.time() - last_time)))
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

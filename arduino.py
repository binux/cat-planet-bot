import time
import queue
import serial
import serial.tools.list_ports

class ArduinoFinger:
  A0 = 14
  YELLOW = A1 = 15
  WHITE = A2 = 16
  RED = A3 = 17
  BLACK = A4 = 18
  BLUE = A5 = 19
  WHITE2 = 8
  RED2 = 9
  BLACK2 = 10
  BLUE2 = 11
  LED = 13

  PRESS_TIME = 50

  def __init__(self, device=None):
    if not device:
      for each in serial.tools.list_ports.comports():
        if (each.manufacturer and 'arduino' in each.manufacturer.lower()
            or 'USB Serial Device' in each.description):
          device = each.device
          break
      else:
        raise Exception("cannot find serial device.")
    self.serial = serial.Serial(device)
    self.serial.readline()  # wait for OK message from arduino

    self.event_history = {}
    self.finger_down = {}
    self.finger_release = {}
    for pin in [*range(self.A0, self.A5 + 1), *range(8, 13 + 1)]:
      self.event_history[pin] = 0
      self.finger_down[pin] = False
      self.finger_release[pin] = 0

  def _send(self, op, pin, blocking=True):
    data = "%s%d\n" % (op, pin)
    self.event_history[pin] = time.time()
    if not blocking:
      self.serial.write_timeout = 0
    self.serial.write(data.encode('utf8'))
    if not blocking:
      self.serial.write_timeout = None

  def press_down(self, pin, blocking=True, autorelease=None):
    if not self.finger_down[pin]:
      self._send('HIG', pin, blocking)
      self.finger_down[pin] = True
    if autorelease:
      self.finger_release[pin] = time.time() + autorelease

  def press_up(self, pin, blocking=True):
    self._send('LOW', pin, blocking)
    self.finger_down[pin] = False
    self.finger_release[pin] = 0

  def press(self, pin):
    self.press_down(pin)
    time.sleep(self.PRESS_TIME / 1000)
    self.press_up(pin)

  def throttle_press(self, pin, timeout=1):
    now = time.time()
    if now - self.event_history[pin] < timeout:
      return
    self.press(pin)

  def autorelease(self, deadline=0.1):
    now = time.time()
    deadline = now + deadline
    todo = sorted([(v, p) for p, v in self.finger_release.items()
            if v and v <= deadline])
    for t, pin in todo:
      if t - now > 0.01:
        time.sleep(t - now)
        now = time.time()
      self.press_up(pin, blocking=False)
    if todo:
      return True
    return False

  def all_press_up(self):
    for pin, pressed in self.finger_down.items():
      if pressed:
        self.press_up(pin)

if __name__ == '__main__':
  finger = ArduinoFinger()
  for i in range(1):
    time.sleep(1)
    finger.press(finger.BLUE)
    for i in range(70):
      time.sleep(0.05)
      finger.press(finger.WHITE)
    time.sleep(2)
    finger.press(finger.RED)
    time.sleep(2)
    for i in range(2):
      finger.press(finger.BLACK)
      time.sleep(0.5)
    time.sleep(5)
    finger.press(finger.ORANGE)

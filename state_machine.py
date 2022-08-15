import time
import collections
import numpy as np
from arduino import ArduinoFinger
from fish_idle_item_detector import FishIdleItemDetector


class StateMachine:
  RED = ArduinoFinger.RED
  BLUE = ArduinoFinger.BLUE
  BLACK = ArduinoFinger.BLACK
  WHITE = ArduinoFinger.WHITE
  ITEM_BUTTON = [ArduinoFinger.YELLOW, ArduinoFinger.ORANGE]

  def __init__(self, ns, get_frame):
    self.ns = ns
    self.get_frame = get_frame

    self.state = 'pause'
    self.last_state = 'pause'
    self.state_args = []
    self.last_state_time = time.time()
    ns.state = self.state

    self.finger = ArduinoFinger()
    self.idle_item_detector = FishIdleItemDetector()

    self.ui_type = None
    self.first_ui_time = 0
    self.frame_latency_history = collections.deque(maxlen=10)
    self.frame_latency = 0
    self.bad_behavior = 0

  def toggle_pause(self):
    if self.state == 'pause':
      self.bad_behavior = 0
      self.change('idle')
    else:
      self.change('pause')

  def change(self, to, *args):
    if self.state != to:
      for key in [
        'on_any_to_%s' % to,
        'on_%s_to_%s' % (self.state, to),
        'on_%s_to_any' % self.state,
      ]:
        if getattr(self, key, None):
          getattr(self, key)(self.state, to)
      self.last_state = self.state
      self.state = to
      self.state_args = args
      self.last_state_time = time.time()
      self.ns.state = to

  @property
  def time_on_ui(self):
    return time.time() - self.first_ui_time

  @property
  def time_on_state(self):
    return time.time() - self.last_state_time

  def reset_state_time(self):
    self.last_state_time = time.time()

  def on_ui_type(self, frame_time, ui_type):
    if self.ui_type != ui_type:
      self.ui_type = ui_type
      self.first_ui_time = frame_time
    if self.bad_behavior >= 3:
      self.change('pause')
    if self.state != 'pause':
      if hasattr(self, 'on_' + ui_type):
        getattr(self, 'on_' + ui_type)()

  def on_fish_drag(self, *args, **kwargs):
    deg = self.detector_data['fish_drag_anchor_deg']
    target_deg = self.detector_data['fish_drag_target_deg']
    if deg is None:
      return

    self.change('drag')
    delta = deg - target_deg
    if delta < 0:
      self.finger.press_up(self.RED)
    else:
      self.finger.press_down(self.RED, autorelease=0.2)

  def on_fish_idle(self, *args, **kwargs):
    if self.state not in ('idle', 'reward', 'shopping', 'drag'):
      return

    if self.time_on_ui > 5 and self.time_on_state > 5:
      self.ns.fish_items = fish_items = self.idle_item_detector.detect(self.get_frame())
      if (not fish_items or any(x for x in fish_items if x is None)):
        if self.time_on_ui > 30:
          self.change('pause')
        return
      for i, item in enumerate(fish_items):
        if item != 0:
          continue
        if i < len(self.ITEM_BUTTON):
          self.finger.press(self.ITEM_BUTTON[i])
          break
        else:
          # no enough item exit
          self.change('pause')
    elif self.time_on_ui > 1:
      self.change('idle')
      self.finger.throttle_press(self.RED, timeout=2)

  def on_any_to_idle(self, *args, **kwargs):
    if self.ui_type == 'fish_idle':
      self.ns.fish_items = self.idle_item_detector.detect(self.get_frame())

  def on_drag_to_idle(self, *args, **kwargs):
    print('State changed from drag to idle, anything wrong?')
    self.bad_behavior += 1

  def on_any_to_cast(self, state, to):
    self.last_cast_ring_radius = 0
    self.last_cast_ring_time = 0
    self.last_cast_speed = 0
    self.speed = collections.deque(maxlen=5)
    self.predicted_cast_radius = 0
    self.speed_when_cast = 0
    self.target_radius = collections.deque(maxlen=100)

  def on_cast_to_any(self, state, to):
    if not self.speed_when_cast:
      return
    frame_latency = (
        (self.last_cast_ring_radius - self.predicted_cast_radius) /
        self.speed_when_cast)
    self.frame_latency_history.append(frame_latency)
    self.frame_latency = np.mean(self.frame_latency_history)
    self.ns.frame_latency = self.frame_latency

    target_radius = np.mean(self.target_radius)
    print('latency: %.1f %.1f %.1f %.1f' %
          (target_radius, self.predicted_cast_radius,
           self.last_cast_ring_radius, self.speed_when_cast))

  def on_fish_ring(self, *args, **kwargs):
    radius = self.detector_data['fish_cast_ring_radius']
    detect_target_radius = self.detector_data['fish_cast_ring_target_radius']
    frame_time = self.detector_data['frame_time']
    if not radius:
      return
    self.change('cast')

    # skip first sample
    if not self.last_cast_ring_time:
      self.last_cast_ring_radius = radius
      self.last_cast_ring_time = frame_time
      return

    # skip turn back from last frame
    predicted_current_pos = (
        self.last_cast_ring_radius +
        (frame_time - self.last_cast_ring_time) * self.last_cast_speed)
    if predicted_current_pos < 0 or predicted_current_pos > 630:
      self.last_cast_ring_radius = radius
      self.last_cast_ring_time = frame_time
      self.last_cast_speed = 0
      return

    # update target ring
    if detect_target_radius:
      self.target_radius.append(detect_target_radius)

    speed = ((radius - self.last_cast_ring_radius) /
             (frame_time - self.last_cast_ring_time))
    if self.speed and np.sign(self.speed[0]) != np.sign(speed):
      self.speed.clear()
    self.speed.append(speed)
    speed = np.mean(self.speed)
    self.last_cast_ring_radius = radius
    self.last_cast_ring_time = frame_time
    self.last_cast_speed = speed

    if len(self.speed) < 3 or len(self.target_radius) < 5:
      return

    target_radius = np.mean(self.target_radius)
    # current actual radius (if we click immediately).
    current_radius = (speed *
                      (time.time() - frame_time + self.frame_latency) +
                      radius)
    next_frame_radius = speed * 0.2 + current_radius
    low, high = sorted([current_radius, next_frame_radius])
    if low <= target_radius <= high:
      t = (target_radius - current_radius) / speed
      if t > 0.005:
        time.sleep(t)
      # predcted_cast_radius shouldn't contain latency fix to update latency
      predicted_cast_radius = (speed * (time.time() - frame_time) + radius)
      self.predicted_cast_radius = predicted_cast_radius
      self.speed_when_cast = speed
      self.finger.press(self.RED)
      self.speed.clear()  # clear speed to prevent click twice


  def on_shopping(self, *args, **kwargs):
    if self.state not in ('idle', 'shopping'):
      return

    self.change('shopping')
    if self.time_on_state < 5:
      self.finger.throttle_press(self.BLACK)
    else:
      self.finger.throttle_press(self.WHITE)

  def on_fish_reward(self, *args, **kwargs):
    if self.time_on_ui > 3:
      self.change('reward')
      self.finger.throttle_press(self.RED)

  def on_mise(self, *args, **kwargs):
    self.finger.throttle_press(self.RED)

  def on_unknown(self, *args, **kwargs):
    self.finger.throttle_press(self.RED)

  def on_not_supported(self, *args, **kwargs):
    self.finger.throttle_press(self.RED)

  def on_fish_bottle(self, *args, **kwargs):
    self.finger.throttle_press(self.BLUE)

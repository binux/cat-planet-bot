import time
import logging
import collections
import numpy as np
from fish_idle_item_detector import FishIdleItemDetector
from arduino import ArduinoFinger
from state_machine import StateMachine


class FishingMachine(StateMachine):
  RED = ArduinoFinger.RED
  BLUE = ArduinoFinger.BLUE
  BLACK = ArduinoFinger.BLACK
  WHITE = ArduinoFinger.WHITE
  ITEM_BUTTON = [ArduinoFinger.YELLOW, ArduinoFinger.ORANGE]

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.idle_item_detector = FishIdleItemDetector()
    self.frame_latency_history = collections.deque(maxlen=10)
    self.frame_latency = 0

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
    elif self.time_on_ui > 1 and self.time_on_state > 1:
      self.change('idle')
      self.finger.throttle_press(self.RED, timeout=2)

  def on_any_to_idle(self, *args, **kwargs):
    if self.ui_type == 'fish_idle':
      self.ns.fish_items = self.idle_item_detector.detect(self.get_frame())

  def on_drag_to_idle(self, *args, **kwargs):
    logging.warning('State changed from drag to idle, anything wrong?')
    self.bad_behavior += 1

  def on_any_to_cast(self, state, to):
    self.cast_ring_history = collections.deque(maxlen=5)
    self.predicted_cast_radius = 0
    self.speed_when_cast = 0
    self.target_radius = collections.deque(maxlen=100)

  def on_cast_to_any(self, state, to):
    if not self.speed_when_cast:
      return
    if not self.cast_ring_history:
      return
    last_ring_radius = np.mean([rad for _, rad in self.cast_ring_history])
    frame_latency = ((last_ring_radius - self.predicted_cast_radius) /
                     self.speed_when_cast)
    if frame_latency < 0:
      return
    self.frame_latency_history.append(frame_latency)
    self.frame_latency = np.mean(self.frame_latency_history)
    self.ns.frame_latency = self.frame_latency

    target_radius = np.mean(self.target_radius)
    logging.debug('latency: %.1f %.1f %.1f %.1f', target_radius,
                  self.predicted_cast_radius, last_ring_radius,
                  self.speed_when_cast)

  def on_fish_ring(self, *args, **kwargs):
    radius = self.detector_data['fish_cast_ring_radius']
    detect_target_radius = self.detector_data['fish_cast_ring_target_radius']
    frame_time = self.detector_data['frame_time']
    if not radius:
      return
    self.change('cast')

    self.cast_ring_history.append((frame_time, radius))
    if detect_target_radius:
      self.target_radius.append(detect_target_radius)

    if len(self.cast_ring_history) != self.cast_ring_history.maxlen:
      return
    if len(self.target_radius) < 5:
      return
    # same direction
    dx = np.diff([rad for _, rad in self.cast_ring_history])
    if not (np.all(dx <= 0) or np.all(dx >= 0)):
      return

    frame_interval = (
        (self.cast_ring_history[-1][0] - self.cast_ring_history[0][0]) /
        len(self.cast_ring_history))
    speed = ((self.cast_ring_history[-1][1] - self.cast_ring_history[0][1]) /
             (self.cast_ring_history[-1][0] - self.cast_ring_history[0][0]))
    if not speed or abs(speed) < 200:
      return

    now = time.time()
    target_radius = np.mean(self.target_radius)
    # current actual radius (if we click immediately).
    current_radius = (speed * (now - frame_time + self.frame_latency) + radius)
    next_frame_radius = speed * 5 * frame_interval + current_radius
    low, high = sorted([current_radius, next_frame_radius])
    if low <= target_radius <= high:
      t = (target_radius - current_radius) / speed
      if t > 0.005:
        time.sleep(t)
      # predicted_cast_radius shouldn't contain latency fix to update latency
      predicted_cast_radius = (speed * (now + t - frame_time) + radius)
      self.predicted_cast_radius = predicted_cast_radius
      self.speed_when_cast = speed
      self.finger.press(self.RED)
      self.cast_ring_history.clear()  # clear speed to prevent click twice


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
    if self.time_on_ui > 1:
      self.finger.throttle_press(self.RED)

  def on_unknown(self, *args, **kwargs):
    if self.time_on_ui > 3:
      self.finger.throttle_press(self.RED)

  def on_not_supported(self, *args, **kwargs):
    if self.time_on_ui > 3:
      self.finger.throttle_press(self.RED)

  def on_fish_bottle(self, *args, **kwargs):
    self.finger.throttle_press(self.BLUE)

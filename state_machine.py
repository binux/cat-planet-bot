import time
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
    self.state_args = []
    self.last_state_time = time.time()
    ns.state = self.state

    self.finger = ArduinoFinger()
    self.idle_item_detector = FishIdleItemDetector()

    self.last_ui = None
    self.last_ui_time = 0
    self.frame_seq = 0
    self.frame_latency = 0

  def toggle_pause(self):
    if self.state == 'pause':
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
      self.state = to
      self.state_args = args
      self.last_state_time = time.time()
      self.ns.state = to

  @property
  def time_on_ui(self):
    return time.time() - self.last_ui_time

  @property
  def time_on_state(self):
    return time.time() - self.last_state_time

  def reset_state_time(self):
    self.last_state_time = time.time()

  def on_ui_type(self, ui_type):
    if self.state == 'pause':
      return
    if hasattr(self, 'on_' + ui_type):
      getattr(self, 'on_' + ui_type)()

  def run(self):
    ns = self.ns
    while not ns.quit:
      # process all pending finger releases
      self.finger.autorelease(0)

      if not ns.bot_command.empty():
        command, args = ns.bot_command.get_nowait()
        getattr(self, command)(*args)

      classifier_data = ns.classifier_data
      if not classifier_data:
        ns.classifier_event.wait()
        continue
      ui_type = classifier_data['ui_type']

      detector_data = ns.detector_data
      if not detector_data:
        ns.detector_event.wait()
        continue

      frame_time = detector_data['frame_time']
      frame_seq = detector_data['frame_seq']
      if frame_seq == self.frame_seq:
        # process finger release during frame waiting
        # otherwise wait for classifier event
        if not self.finger.autorelease(0.1):
          ns.detector_event.wait(0.1)
        continue

      self.frame_seq = frame_seq
      self.detector_data = detector_data
      if self.last_ui != ui_type:
        self.last_ui = ui_type
        self.last_ui_time = frame_time

      self.on_ui_type(ui_type)
      ns.processing_event_queue.put((frame_seq, 'bot', time.time()))
    self.finger.all_press_up()

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
      self.finger.press_down(self.RED)

  def on_fish_idle(self, *args, **kwargs):
    if self.state in ('idle', 'reward'):
      if self.time_on_ui > 1 and self.time_on_state > 3:
        self.finger.throttle_press(self.RED)
        self.reset_state_time()
    elif self.state in ('shopping', ):
      if self.time_on_ui > 1 and self.time_on_state > 1:
        self.finger.throttle_press(self.state_args[0])
        self.reset_state_time()
    elif self.time_on_state > 5:
      fish_item = self.idle_item_detector.detect(self.get_frame())
      if not fish_item or any(x for x in fish_item if x is None):
        self.change('idle')
        return
      for i, item in enumerate(fish_item):
        if item != 0:
          continue
        if i < len(self.ITEM_BUTTON):
          self.finger.press(self.ITEM_BUTTON[i])
          self.change('shopping', self.ITEM_BUTTON[i])
          break
        else:
          # no enough item exit
          self.change('pause')

  def on_any_to_cast(self, state, to):
    self.last_cast_ring_radius = 0
    self.last_cast_ring_time = 0
    self.predicted_cast_radius = 0
    self.speed_when_cast = 0

  def on_cast_to_any(self, state, to):
    if not self.speed_when_cast:
      return
    self.frame_latency = (
        (self.predicted_cast_radius - self.last_cast_ring_radius) /
        self.speed_when_cast)
    self.ns.frame_latency = self.frame_latency

  def on_fish_ring(self, *args, **kwargs):
    radius = self.detector_data['fish_cast_ring_radius']
    target_radius = self.detector_data['fish_cast_ring_target_radius']
    frame_time = self.detector_data['frame_time']
    if not radius:
      return

    self.change('cast')
    if self.last_cast_ring_time:
      speed = ((radius - self.last_cast_ring_radius) /
               (frame_time - self.last_cast_ring_time))
      current_radius = speed * (time.time() - frame_time) + radius
      next_frame_radius = speed * 0.1 + radius  # assume 10 FPS
      low, high = sorted([current_radius, next_frame_radius])
      # if target_radius - 10 < current_radius < target_radius + 10:
      #   self.finger.press(self.RED)
      print('cast:', frame_time, radius, current_radius, next_frame_radius, speed)
      if low <= target_radius <= high:
        t = (target_radius - current_radius) / speed - self.frame_latency
        # print('t=', t)
        if t > 0.02:
          time.sleep(t)
        self.finger.press(self.RED)
        self.predicted_cast_radius = (speed * (time.time() - frame_time) +
                                      radius)
        self.speed_when_cast = speed
        print('casted:', self.predicted_cast_radius, self.speed_when_cast)

    self.last_cast_ring_radius = radius
    self.last_cast_ring_time = frame_time

  def on_shopping(self, *args, **kwargs):
    if self.state == 'shopping':
      if self.time_on_state < 5:
        self.finger.throttle_press(self.BLACK)
      else:
        self.finger.throttle_press(self.WHITE)

  def on_fish_reward(self, *args, **kwargs):
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

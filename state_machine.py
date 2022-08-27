import time
from arduino import ArduinoFinger

class StateMachine:
  BAD_BEHAVIOR_PAUSE = 3

  def __init__(self, ns, get_frame):
    self.ns = ns
    self.get_frame = get_frame

    self.state = 'pause'
    self.last_state = 'pause'
    self.state_args = []
    self.last_state_time = time.time()
    ns.state = self.state

    self.finger = ArduinoFinger()

    self.ui_type = None
    self.first_ui_time = 0
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
    if self.bad_behavior >= self.BAD_BEHAVIOR_PAUSE:
      self.change('pause')
    if self.state != 'pause':
      if hasattr(self, 'on_' + ui_type):
        getattr(self, 'on_' + ui_type)()

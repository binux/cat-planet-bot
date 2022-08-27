import time
import logging
from state_machine import StateMachine
from arduino import ArduinoFinger


class GraffitiMachine(StateMachine):
  CHARACTER_AUTO_SELECT = START = ArduinoFinger.RED
  CHARACTER_SELECT = SIDE_BATTLE = ArduinoFinger.BLUE
  MAIN_BATTLE = ArduinoFinger.WHITE
  CHARACTER_CONFIRM = BATTLE_EXIT = BATTLE_START = ArduinoFinger.BLUE2
  CHARACTERS = [
      ArduinoFinger.YELLOW,
      ArduinoFinger.BLACK2,
      ArduinoFinger.WHITE2,
      ArduinoFinger.RED2,
  ]
  CHARACTER_ORDER = [4, 3, 2, 1]
  BAD_BEHAVIOR_PAUSE = 30

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.graffiti_step = 1

  def on_graffiti_select(self, *args, **kwargs):
    if self.time_on_state < 1:
      return
    self.graffiti_step = 1
    self.finger.throttle_press(self.START)
    self.change('select')

  def on_graffiti_step(self, *args, **kwargs):
    if self.time_on_state < 1:
      return
    if self.state in ('step', ) and self.time_on_state > 5:
      self.graffiti_step += 1
      self.graffiti_step %= 2
      self.change('select')
      return
    if self.graffiti_step == 0:
      self.finger.throttle_press(self.SIDE_BATTLE)
      self.change('step')
    elif self.graffiti_step == 1:
      self.finger.throttle_press(self.MAIN_BATTLE)
      self.change('step')

  def on_battle_prepare(self, *args, **kwargs):
    if self.time_on_state < 1:
      return
    if self.state == 'battle_prepare':
      if self.time_on_state > 5:
        self.change('step')
        logging.warning('Start battle failed.')
        self.bad_behavior += 1
      return
    if self.state == 'character_selection':
      self.finger.press(self.BATTLE_START)
      self.change('battle_prepare')
    elif self.graffiti_step == 0:
      self.finger.press(self.CHARACTER_AUTO_SELECT)
      time.sleep(1)
      self.finger.press(self.BATTLE_START)
      self.change('battle_prepare')
    elif self.graffiti_step == 1:
      self.finger.press(self.CHARACTER_AUTO_SELECT)
      time.sleep(1)
      self.finger.press(self.CHARACTER_AUTO_SELECT)
      time.sleep(1)
      self.finger.press(self.CHARACTER_SELECT)
      time.sleep(1)
      self.change('battle_prepare')

  def on_character_selection(self, *args, **kwargs):
    if self.time_on_state < 1:
      return
    if self.state == 'character_selection' and self.time_on_state > 5:
      self.change('battle_prepare')
      return
    for i in self.CHARACTER_ORDER:
      self.finger.press(self.CHARACTERS[i-1])
      time.sleep(0.5)
    for i in self.CHARACTER_ORDER:
      self.finger.press(self.CHARACTERS[i-1])
      time.sleep(0.5)
    self.finger.press(self.CHARACTER_CONFIRM)
    self.change('character_selection')

  def on_battle(self, *args, **kwargs):
    if self.time_on_state < 1:
      return
    self.change('battle')

  def on_battle_result_success(self, *args, **kwargs):
    if self.time_on_state < 1:
      return
    if self.state in ('battle_success', ):
      self.finger.throttle_press(self.BATTLE_EXIT)
    else:
      self.finger.throttle_press(self.BATTLE_EXIT)
      self.graffiti_step += 1
      self.change('battle_success')

  def on_any_to_battle_success(self, *args, **kargs):
    if self.graffiti_step == 2:
      logging.info('graffiti success!')

  def on_battle_result_failed(self, *args, **kwargs):
    if self.time_on_state < 5:
      return
    if self.state in ('battle_failed', ):
      self.finger.throttle_press(self.BATTLE_EXIT)
    else:
      logging.warning('battle failed at step %d' % self.graffiti_step)
      self.bad_behavior += 1
      self.finger.throttle_press(self.BATTLE_EXIT)
      self.change('battle_failed')

  def on_mise(self, *args, **kwargs):
    if self.time_on_state < 1:
      return
    self.finger.throttle_press(self.BATTLE_EXIT)

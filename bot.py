import argparse
import collections
import time
import datetime
import multiprocessing
import sys
import logging
import numpy as np
from multiprocessing import shared_memory

frame_buffer_size = 1140 * 1546 * 3 * 120
frame_buffer_dtype = np.uint8
options = None

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname).1s %(asctime)s %(module)s:%(lineno)d] %(message)s')

def get_frame(frame_shm, frame_shape, offset):
  buf = np.frombuffer(frame_shm.buf, frame_buffer_dtype, np.prod(frame_shape),
                      offset)
  return buf.reshape(frame_shape)

def frame(ns, frame_shm):
  # from screen_capture import ScreenCapture
  from obs_capture import OBSCapture as Capture

  ns.reset_windows_pos = False

  screen = Capture()
  frame_seq = 0
  frame_offset = 0
  frame_buffer = np.ndarray(frame_buffer_size,
                            frame_buffer_dtype,
                            buffer=frame_shm.buf)
  ns.frame_ready = True

  for frame in screen.capture():
    if ns.quit:
      break
    frame_seq += 1
    now = time.time()
    if frame.size + frame_offset > frame_buffer.size:
      frame_offset = 0
    frame_buffer[frame_offset:frame_offset + frame.size] = frame.flatten()
    ns.frame_data = dict(
        frame_offset=frame_offset,
        frame_shape=frame.shape,
        frame_seq=frame_seq,
        frame_time=now,
    )
    frame_offset += frame_offset + frame.size
    ns.processing_event_queue.put((frame_seq, now, 'frame', now))
    ns.frame_event.set()
    ns.frame_event.clear()
    if ns.reset_windows_pos:
      ns.reset_windows_pos = False
      screen.reset_windows_pos()

def classifier(ns, frame_shm):
  import cv2 as cv
  from classifier import CatCityClassifier

  classifier = CatCityClassifier()
  ns.classifier_ready = True

  # wait for first frame
  ns.frame_event.wait()
  processed_seq = 0
  last_save = 0
  while not ns.quit:
    # get last frame
    frame_data = ns.frame_data
    frame_seq, frame_time, frame_shape, frame_offset = [
        frame_data[k]
        for k in 'frame_seq, frame_time, frame_shape, frame_offset'.split(', ')
    ]
    if frame_seq - processed_seq < 10:
      continue
    frame = get_frame(frame_shm, frame_shape, frame_offset)
    if frame_seq == processed_seq:
      ns.frame_event.wait(0.1)
      continue
    processed_seq = frame_seq

    # predict
    ui_type, score = classifier.predict(frame)
    ns.classifier_data = dict(
        ui_type=ui_type,
        score=score,
        frame_seq=frame_seq,
        frame_time=frame_time,
    )
    ns.processing_event_queue.put((frame_seq, frame_time, 'classifier', time.time()))
    ns.classifier_event.set()
    ns.classifier_event.clear()

    # save screenshot
    if ((ui_type == 'unknown' or score < 70) and ns.save_screenshot
        and time.time() - last_save > 1):
      filename = ('data/raw/%s.png' %
                  datetime.datetime.now().isoformat().replace(':', '.'))
      cv.imwrite(filename, frame)
      last_save = time.time()

def detector(ns, frame_shm, debug=False):
  from fish_drag_anchor_detection import AnchorDetector
  from fish_cast_ring_detector import FishCastRingDetector

  anchor_detector = AnchorDetector(debug=debug)
  cast_ring_detector = FishCastRingDetector(debug=debug)
  ns.detector_ready = True

  # wait for first event
  ns.classifier_event.wait()
  processed_seq = 0
  while not ns.quit:
    # get last frame
    frame_data = ns.frame_data
    frame_seq, frame_time, frame_shape, frame_offset = [
        frame_data[k]
        for k in 'frame_seq, frame_time, frame_shape, frame_offset'.split(', ')
    ]
    frame = get_frame(frame_shm, frame_shape, frame_offset)
    if frame_seq == processed_seq:
      ns.frame_event.wait(0.1)
      continue
    processed_seq = frame_seq

    # get ui_type
    classifier_data = ns.classifier_data
    ui_type = classifier_data['ui_type']

    fish_drag_anchor_deg = None
    if ui_type == 'fish_drag':
      fish_drag_anchor_deg = anchor_detector.detect(frame)
    fish_cast_ring_target_radius = None
    fish_cast_ring_radius = None
    if ui_type == 'fish_ring':
      fish_cast_ring_target_radius, fish_cast_ring_radius = cast_ring_detector.detect(frame)

    ns.detector_data = dict(
        fish_cast_ring_radius=fish_cast_ring_radius,
        fish_cast_ring_target_radius=fish_cast_ring_target_radius,
        fish_drag_anchor_deg=fish_drag_anchor_deg,
        fish_drag_target_deg=anchor_detector.target_deg,
        frame_seq=frame_seq,
        frame_time=frame_time,
    )
    ns.processing_event_queue.put((frame_seq, frame_time, 'detector', time.time()))
    ns.detector_event.set()
    ns.detector_event.clear()


def bot(ns, frame_shm):
  if options.script == 'fishing':
    from fishing_machine import FishingMachine as StateMachine
  elif options.script == 'graffiti':
    from graffiti_machine import GraffitiMachine as StateMachine

  ns.frame_latency = 0
  ns.fish_items = None

  def get_frame_fn():
    frame_data = ns.frame_data
    frame_shape, frame_offset = [
        frame_data[k]
        for k in 'frame_shape, frame_offset'.split(', ')
    ]
    return get_frame(frame_shm, frame_shape, frame_offset)

  sm = StateMachine(ns, get_frame=get_frame_fn)
  ns.classifier_event.wait()
  ns.bot_ready.set()

  if not ns.detector_data:
    ns.detector_event.wait()
  if not ns.classifier_data:
    ns.classifier_event.wait()

  processed_seq = 0
  while not ns.quit:
    # process all pending finger releases
    sm.finger.autorelease(0)
    # process all bot command
    while not ns.bot_command.empty():
      command, args = ns.bot_command.get_nowait()
      getattr(sm, command)(*args)

    detector_data = ns.detector_data
    frame_time = detector_data['frame_time']
    frame_seq = detector_data['frame_seq']
    if frame_seq == processed_seq:
      ns.detector_event.wait(0.1)
      continue
    processed_seq = frame_seq
    sm.detector_data = detector_data

    classifier_data = ns.classifier_data
    ui_type = classifier_data['ui_type']
    sm.on_ui_type(frame_time, ui_type)

    ns.processing_event_queue.put((frame_seq, frame_time, 'bot', time.time()))

  sm.finger.all_press_up()

def get_options():
  parser = argparse.ArgumentParser(description='Bot launcher.')
  parser.add_argument('-d', '--debug', help='Debug', action='store_true')
  parser.add_argument('script',
                      help='Bot Script',
                      choices=['fishing', 'graffiti'],
                      default='graffiti')
  return parser.parse_args()

def main(frame_shm, debug=False):
  import cv2 as cv

  manager = multiprocessing.Manager()
  ns = manager.Namespace()
  ns.quit = False
  ns.save_screenshot = False
  ns.processing_event_queue = manager.Queue()
  ns.frame_ready = False
  ns.frame_event = manager.Event()
  ns.frame_data = None
  ns.classifier_ready = False
  ns.classifier_event = manager.Event()
  ns.classifier_data = None
  ns.detector_ready = False
  ns.detector_event = manager.Event()
  ns.detector_data = None
  ns.bot_ready = manager.Event()
  ns.bot_command = manager.Queue()

  processes = [
    multiprocessing.Process(target=frame, args=(ns, frame_shm)),
    multiprocessing.Process(target=classifier, args=(ns, frame_shm)),
    multiprocessing.Process(target=detector, args=(ns, frame_shm, debug)),
    multiprocessing.Process(target=bot, args=(ns, frame_shm)),
  ]
  for p in processes:
    p.start()

  def on_key_code(c):
    if c == 'q':
      ns.quit = True
    elif c == 's':
      frame_data = ns.frame_data
      frame_shape, frame_offset = [
          frame_data[k]
          for k in 'frame_shape, frame_offset'.split(', ')
      ]
      frame = get_frame(frame_shm, frame_shape, frame_offset)
      filename = ('data/raw/%s.png' %
                  datetime.datetime.now().isoformat().replace(':', '-'))
      cv.imwrite(filename, frame)
      ns.save_screenshot = not ns.save_screenshot
    elif c == 'r':
      ns.reset_windows_pos = True
    elif c == 'k':
      processes[-1].kill()
      processes[-1] = multiprocessing.Process(target=bot, args=(ns, frame_shm))
      processes[-1].start()
    elif c == 'o':
      if video_out[0]:
        video_out[0].release()
        video_out[0] = None
      else:
        frame_shape = ns.frame_data['frame_shape']
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        video_out[0] = cv.VideoWriter(
            'data/video/%s.mp4' %
            datetime.datetime.now().isoformat().replace(':', '-'), fourcc,
            30.0, (frame_shape[1], frame_shape[0]))
    elif c == 'p':
      ns.bot_command.put(('toggle_pause', []))

  video_out = [None]
  blank_frame = np.zeros((600, 1200, 3), np.uint8)
  cv.namedWindow("OpenCV", cv.WINDOW_NORMAL)
  def update_info_cv(texts):
    if debug and ns.frame_data:
      frame_data = ns.frame_data
      frame_shape, frame_offset = [
          frame_data[k]
          for k in 'frame_shape, frame_offset'.split(', ')
      ]
      frame = get_frame(frame_shm, frame_shape, frame_offset)
    else:
      frame = blank_frame
      frame[:] = 0
    x0, y0 = 10, 100
    font_face, font_scale, thickness = cv.FONT_HERSHEY_SIMPLEX, 1, 2
    (xd, yd), baseline = cv.getTextSize("F", font_face, font_scale, thickness)
    for i, line in enumerate(texts):
      cv.putText(frame,
                 str(line), (x0, y0 + i * (yd + baseline)),
                 font_face,
                 font_scale,
                 (0, 255, 0),
                 thickness=thickness)
    if video_out[0]:
      video_out[0].write(frame)
    cv.imshow("OpenCV", frame)
    key_code = cv.waitKey(1) & 0xFF
    if key_code:
      on_key_code(chr(key_code))

  def update_info():
    texts = []
    state_message = []
    if ns.bot_ready.is_set():
      state_message.append("%s " % (ns.state))
    if ns.save_screenshot:
      state_message.append("💾")
    if video_out[0]:
      state_message.append("📹")
    texts.append(''.join(state_message))
    fps, latency = get_fps_latency()
    texts.append("FPS: %04.1f|%04.1f|%04.1f|%04.1f" % (
        fps.get('frame', 0),
        fps.get('classifier', 0),
        fps.get('detector', 0),
        fps.get('bot', 0),
    ))
    texts.append("Latency: %02.0f|%02.0f|%02.0f|%02.0f" % (
        ns.frame_latency * 1000 if ns.bot_ready.is_set() else 0,
        latency.get('classifier', 0),
        latency.get('detector', 0),
        latency.get('bot', 0),
    ))
    if ns.bot_ready.is_set():
      texts.append("items = %r" % ns.fish_items)
    classifier_data = ns.classifier_data
    if classifier_data:
      texts.append("%s = %03d%%" %
                   (classifier_data['ui_type'], classifier_data['score']))
    detector_data = ns.detector_data
    if detector_data:
      fish_drag_anchor_deg = detector_data['fish_drag_anchor_deg']
      fish_drag_target_deg = detector_data['fish_drag_target_deg']
      if fish_drag_anchor_deg is not None:
        texts.append("anchor: %+05.1f %04.1f" % (
            fish_drag_target_deg - fish_drag_anchor_deg,
            fish_drag_anchor_deg,
        ))
      fish_cast_ring_radius = detector_data['fish_cast_ring_radius']
      fish_cast_ring_target_radius = detector_data['fish_cast_ring_target_radius']
      if fish_cast_ring_radius is not None:
        texts.append("ring:  %+06.1f %05.1f" % (
            fish_cast_ring_target_radius - fish_cast_ring_radius
            if fish_cast_ring_radius and fish_cast_ring_target_radius else -1,
            fish_cast_ring_radius,
        ))
    update_info_cv(texts)

  deques = {}
  def process_event_queue():
    queue = ns.processing_event_queue
    while not queue.empty():
      frame_seq, frame_time, worker, process_time = queue.get()
      if worker not in deques:
        deques[worker] = collections.deque(maxlen=100)
      deques[worker].append((process_time, process_time - frame_time))

  last_fps = [0, None]
  def get_fps_latency():
    if 'frame' not in deques:
      return {}, {}
    if time.time() - last_fps[0] < 0.5:
      return last_fps[1]
    fps = {}
    latency = {}
    for worker, deque in deques.items():
      if len(deque) <= 1:
        continue
      fps[worker] = (len(deque) - 1) / (deque[-1][0] - deque[0][0])
      latency[worker] = np.mean([x[1] for x in deque]) * 1000
    last_fps[:] = [time.time(), (fps, latency)]
    return fps, latency

  while not ns.quit:
    ns.frame_event.wait()
    process_event_queue()
    update_info()

    for p in processes:
      if not p.is_alive():
        ns.quit = True

  for p in processes:
    p.join(1)
    p.terminate()

if __name__ == '__main__':
  options = get_options()
  dsize = np.dtype(frame_buffer_dtype).itemsize * frame_buffer_size
  frame_shm = shared_memory.SharedMemory(create=True, size=dsize)
  try:
    main(frame_shm, debug=options.debug)
  finally:
    frame_shm.unlink()
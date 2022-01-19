import pystk
import numpy as np
import matplotlib.pyplot as plt

class PyTuxActionCritic:
  _singleton = None
  RESCUE_TIMEOUT = 30

  def __init__(self, screen_width=128, screen_height=96, steps=None, verbose=False):
      assert PyTuxActionCritic._singleton is None, "Cannot create more than one pytux object"
      PyTuxActionCritic._singleton = self
      self.config = pystk.GraphicsConfig.hd()
      self.config.screen_width = screen_width
      self.config.screen_height = screen_height
      pystk.init(self.config)
      self.k = None
      self.t = 0
      self.state = None
      self.track = None
      self.last_rescue = 0
      self.distance = 0
      self.steps = steps
      self.fig = None
      self.ax = None
      self.verbose = verbose
      self.last_frame = None
      self.termination_steps = 0
      self.max_distance = 0

      self.last_point = None
      self.last_aim_point = None

      if verbose:
            self.fig, self.ax = plt.subplots(1, 1)

  def rgb2gray(self, rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

  @staticmethod
  def _point_on_track(distance, track, offset=0.0):
      """
      Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
      Returns a 3d coordinate
      """
      node_idx = np.searchsorted(track.path_distance[..., 1],
                                  distance % track.path_distance[-1, 1]) % len(track.path_nodes)
      d = track.path_distance[node_idx]
      x = track.path_nodes[node_idx]
      t = (distance + offset - d[0]) / (d[1] - d[0])
      return x[1] * t + x[0] * (1 - t)

  @staticmethod
  def _to_image(x, proj, view):
      p = proj @ view @ np.array(list(x) + [1])
      return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

  def reset(self, track, location=None):
    self.state = pystk.WorldState()
    self.track = pystk.Track()

    self.last_rescue = 0
    self.t = 0
    self.distance = 0
    self.max_distance = 0
    self.termination_steps = 0

    if self.k is not None and self.k.config.track == track:
      self.k.restart()
      self.k.step()
    else:
      if self.k is not None:
          self.k.stop()
          del self.k
      config = pystk.RaceConfig(num_kart=1, laps=1,track=track)
      config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

      self.k = pystk.Race(config)
      self.k.start()
      self.k.step()

    # self.state = pystk.WorldState()
    # self.track = pystk.Track()

    # self.last_frame = np.array(self.k.render_data[0].image)
    self.state.update()
    self.track.update()

    kart = self.state.players[0].kart
    # state = kart.location + kart.velocity + [kart.distance_down_track]
    state = self.rgb2gray(np.array(self.k.render_data[0].image).astype("float32")) / 255.0
    # state = np.array(self.k.render_data[0].instance)
    # state = np.right_shift(state, 10 * np.ones_like(state))
    # state = state / 81920
    state = state.astype("float32")

    if (location is not None):
      self.state.set_kart_location(0, location)

    state = state.flatten()
    state = np.append(state, kart.location + kart.velocity + [kart.distance_down_track] + [self.max_distance])

    # state = np.array(self.k.render_data[0].image).astype("float32")

    self.last_point = np.array(kart.location)

    self.last_aim_point = self._point_on_track(0, self.track)

    return np.array(state, dtype=np.float32)
    # return np.array(self.k.render_data[0].image)

  def set_location(self, location):
    self.state.set_kart_location(0, location)

  def getState(self):
    if (self.k is not None):
      yield np.array(self.k.render_data[0].image)

    yield np.zeros((self.config.screen_height, 
                     self.config.screen_width, 3))
    
  def step(self, action, verbose=False):
    """
    Play a level (track) for a single round.
    :param track: Name of the track
    :param controller: low-level controller, see controller.py
    :param max_frames: Maximum number of frames to play for
    :param verbose: Should we use matplotlib to show the agent drive?
    :return: state, reward, done, time
    """
    # action.brake = False
    # action.rescue = False

    reward = 0
  
    self.state.update()
    self.track.update()

    kart = self.state.players[0].kart

    collided = False

    # im = self.k.render_data[0].image

    proj = np.array(self.state.players[0].camera.projection).T
    view = np.array(self.state.players[0].camera.view).T
    WH2 = np.array([self.config.screen_width, self.config.screen_height]) / 2
    # aim_point_image = self._to_image(aim_point_world, proj, view)
    # loc = self._to_image(kart.location, proj, view)
    # print(" ", np.linalg.norm([kart.location, aim_point_world]))
    # print(" ", WH2*(1+self._to_image(kart.location, proj, view)))

    current_distance = kart.distance_down_track

    # state = kart.location + kart.velocity + [kart.distance_down_track]
    # state = np.array(self.k.render_data[0].instance)
    # state = np.right_shift(state, 10 * np.ones_like(state))
    # state = state / 81920
    # state = state.astype("float32")

    state = self.rgb2gray(np.array(self.k.render_data[0].image).astype("float32")) / 255.0
    im = state
    state = state.flatten()
    state = np.append(state, kart.location + kart.velocity + [kart.distance_down_track] + [self.max_distance])

    # state = np.array(im).astype("float32")

    # if (current_distance - self.distance > 1000):
    #   current_distance -= self.track.length

    aim_point_world = self._point_on_track(current_distance+30, self.track)

    if np.isclose(kart.overall_distance / self.track.length, 1.0, atol=2e-3):
        # if self.verbose:
        print("Finished at t=%d" % self.t)
        reward = 100
        # return np.array(im), reward, True, current_distance # reward for finish
        return np.array(state, dtype=np.float32), reward, True, current_distance # reward for finish

    current_vel = np.linalg.norm(kart.velocity)

    if current_vel < 1.0 and self.t - self.last_rescue > PyTuxActionCritic.RESCUE_TIMEOUT:
        self.last_rescue = self.t
        action.rescue = True
        reward -= 3
        collided = True
        # return np.array(im), reward, False, current_distance

    if self.verbose:
      self.ax.clear()
      self.ax.add_artist(plt.Circle(WH2*(1+self._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))
      self.ax.add_artist(plt.Circle(WH2*(1+self._to_image(aim_point_world, proj, view)), 2, ec='r', fill=False, lw=1.5))

      self.ax.imshow(im)
      # inst = np.array(self.k.render_data[0].instance)
      # inst = np.right_shift(inst, 10 * np.ones_like(inst))
      # inst = inst / 81920
      # inst[inst!=0.4] = 1
      # print(np.unique(inst))
      # self.ax.imshow(inst)
      plt.pause(1e-3)

    self.k.step(action)
    self.t += 1
    # else:
    #   reward += 0.0001

    if (current_distance <= self.max_distance or current_distance - self.max_distance > 300):
      self.termination_steps += 1
      # reward = -1
    else:
      self.termination_steps = 0

    # # 0.4 track

    if (self.termination_steps == 1000):
      return np.array(state, dtype=np.float32), reward, True, current_distance

    if (current_distance > self.distance and current_distance - self.distance < 300 and not collided):
      reward += current_distance - self.distance

    if (current_distance > self.max_distance and current_distance - self.max_distance < 300):
      self.max_distance = current_distance

    self.distance = current_distance

    # return np.array(im), reward, False, current_distance# penalty for each additional step
    return np.array(state, dtype=np.float32), reward, False, current_distance # penalty for each additional step

  def close(self):
    """
    Call this function, once you're done with PyTux
    """
    if self.k is not None:
        self.k.stop()
        del self.k
    pystk.clean()

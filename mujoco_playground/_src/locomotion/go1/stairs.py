# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Stair climbing task for Unitree Go1."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

# from mujoco_playground._src import gait # This import seems to be missing in your provided Go1 context, assuming it's available.
# If not, the 'feet_phase' reward will need to be removed or reimplemented.
# For this example, I will assume a gait module is available or 'feet_phase' is set to 0.

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import go1_constants as consts


def default_config() -> config_dict.ConfigDict:
  """Returns the default configuration for the Go1 stair climbing task."""
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,  # Go1 typically uses 0.004 sim_dt
      episode_length=200,
      Kp=35.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=0.5,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gravity=0.05,
              linvel=0.1,
              gyro=0.2,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking related rewards.
              tracking_lin_vel=2.0,  # High reward for moving forward
              tracking_ang_vel=0.75, # Reward for staying straight
              # Base related rewards.
              lin_vel_z=0.0,         # NO penalty for vertical velocity
              ang_vel_xy=-0.05,      # Small penalty for pitch/roll velocity
              orientation=-1.0,      # Small penalty for non-flat orientation
              base_height=0.0,
              # Energy related rewards.
              torques=-0.0002,
              action_rate=-0.01,
              energy=-0.001,
              dof_acc=0.0,
              # Feet related rewards.
              feet_clearance=0.0,
              feet_air_time=0.1,
              feet_slip=-0.1,
              feet_height=-0.2,
              feet_phase=0.0,  # Set to 0 as gait module is not confirmed
              # Other rewards.
              alive=1.0,
              stand_still=-0.5,
              termination=-10.0,
              collision=-0.1,
              contact_force=0.0, # Set to 0, function is removed
              # Pose related rewards.
              dof_pos_limits=-1.0,
              pose=-0.1,
          ),
          tracking_sigma=0.25,
          max_foot_height=0.15,
          base_height_target=0.27, # Target height from Go1 'home' pose
          max_contact_force=150.0, # Unused, but kept for structure
      ),
      push_config=config_dict.create(
          enable=False,  # Disable pushing for stair climbing
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 2.0],
      ),
      command_config=config_dict.create(
          # These are now overridden by sample_command, but kept for structure
          a=[0.7, 0.0, 0.0],
          b=[1.0, 0.0, 0.0],
      ),
      lin_vel_x=[0.7, 0.7],  # Fixed target velocity
      lin_vel_y=[0.0, 0.0],
      ang_vel_yaw=[0.0, 0.0],
      impl="jax",
      nconmax=100 * 8192,
      njmax=12 + 100 * 4,
  )


class Stairs(go1_base.Go1Env):
  """Go1 environment for climbing stairs.

  This class follows the explicit structure of the G1 Joystick environment
  (defining step, _get_obs, _get_reward, etc.) but is adapted for the
  Go1 quadruped and the specific task of stair climbing.
  """

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        # Hard-code the stairs XML path
        xml_path=consts.FEETONLY_STAIRS_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    """Initializes robot-specific attributes."""
    # --- Adapted from Go1 Script ---
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # Note: First joint is freejoint. (12 DoF)
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    
    # Get the IMU site ID from the base Go1 constants
    # self._torso_imu_site_id = self._mj_model.site(consts.IMU_SITE).id

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)

    # self._feet_floor_found_sensor is inherited from go1_base.Go1Env
    # No need to redefine it here.

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Resets the environment to an initial state."""
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # Randomize initial pose slightly
    # x, y=U(-0.1, 0.1), yaw=U(-0.2, 0.2).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.1, maxval=0.1)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-0.2, maxval=0.2)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # qpos[7:] (12 joints)
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(
        qpos[7:] * jax.random.uniform(key, (12,), minval=0.8, maxval=1.2)
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:], # Use default pose for initial ctrl
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Command is fixed for this task
    cmd = self.sample_command(rng)

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(4), # 4 feet
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "push": jp.array([0.0, 0.0]), # Unused, but kept for structure
        "push_step": 0,
        "push_interval_steps": 10000, # Effectively disabled
        "phase": jp.zeros(2), # Unused if gait reward is 0
        "phase_dt": jp.zeros(1), # Unused if gait reward is 0
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    # Use the inherited _feet_floor_found_sensor
    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])
    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Computes one environment step."""
    # Push logic is disabled in config, so we skip it.
    data = state.data

    # Apply action
    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    # Get contact state (using inherited _feet_floor_found_sensor)
    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    # Get observation, done, and rewards
    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items() if k in self._config.reward_config.scales
    }
    reward = sum(rewards.values()) * self.dt

    # Update state info
    state.info["step"] += 1
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    
    # Command is fixed, no resampling needed.
    # state.info["command"] = self.sample_command(cmd_rng)

    # Reset step counter if done
    state.info["step"] = jp.where(
        done | (state.info["step"] > self._config.episode_length),
        0,
        state.info["step"],
    )
    
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    """Determines if the episode should terminate."""
    # Terminate if torso is too tilted (fall)
    # Use get_upvector as in the reference Go1 Joystick file
    fall_termination = (self.get_upvector(data)[-1] < 0.0)
    
    # Terminate on NaN
    nan_termination = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    return fall_termination | nan_termination

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    """Returns the observation vector."""
    # Use Go1 base functions (no "torso"/"pelvis" arg)
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    # Use get_gravity from Go1 base
    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    # Simplified phase (unused if reward is 0)
    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])

    # Use get_local_linvel from Go1 base
    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        info["command"],  # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,  # 12
        info["last_act"],  # 12
        phase, # 2
    ])

    # Use Go1 base functions
    accelerometer = self.get_accelerometer(data)
    global_angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    root_height = data.qpos[2]

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        global_angvel,  # 3
        joint_angles - self._default_pose, # 12
        joint_vel, # 12
        root_height,  # 1
        data.actuator_force,  # 12
        contact,  # 4
        feet_vel,  # 4*3 = 12
        info["feet_air_time"],  # 4
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    """Returns a dictionary of reward components."""
    del metrics  # Unused.
    
    # Use Go1 base functions (no "torso"/"pelvis" arg)
    base_linvel = self.get_local_linvel(data)
    base_global_linvel = self.get_global_linvel(data)
    base_gyro = self.get_gyro(data)
    base_global_angvel = self.get_global_angvel(data)
    base_upvector = self.get_upvector(data) # Use get_upvector for orientation
    
    return {
        # Tracking rewards.
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], base_linvel
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], base_gyro
        ),
        # Base-related rewards.
        "lin_vel_z": self._cost_lin_vel_z(base_global_linvel),
        "ang_vel_xy": self._cost_ang_vel_xy(base_global_angvel),
        "orientation": self._cost_orientation(base_upvector), # Pass upvector
        "base_height": self._cost_base_height(data.qpos[2]),
        # Energy related rewards.
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "dof_acc": self._cost_dof_acc(data.qacc[6:]),
        # Feet related rewards.
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data, info),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "feet_phase": jp.array(0.0), # Disabled
        # Other rewards.
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "collision": jp.array(0.0), # Disabled
        "contact_force": jp.array(0.0), # Disabled
        # Pose related rewards.
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "pose": self._cost_pose(data.qpos[7:]),
    }

  # REMOVED _cost_contact_force function as Go1 XML lacks force sensors

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    """Penalizes deviation from the default pose."""
    return jp.sum(jp.square(qpos - self._default_pose))

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    """Penalizes being near joint limits."""
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    """Rewards tracking the linear velocity command."""
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    """Rewards tracking the angular velocity command."""
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Base-related rewards.

  def _cost_lin_vel_z(
      self,
      global_linvel_base: jax.Array,
  ) -> jax.Array:
    """Penalizes vertical velocity (z)."""
    # Note: Config sets scale to 0.0, so this has no effect.
    return jp.square(global_linvel_base[2])

  def _cost_ang_vel_xy(self, global_angvel_base: jax.Array) -> jax.Array:
    """Penalizes body roll/pitch velocity."""
    return jp.sum(jp.square(global_angvel_base[:2]))

  def _cost_orientation(self, torso_upvector: jax.Array) -> jax.Array:
    """Penalizes non-flat body orientation."""
    # Match logic from Go1 Joystick: penalize x/y components of up-vector
    return jp.sum(jp.square(torso_upvector[:2]))

  def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
    """Penalizes deviation from target base height."""
    return jp.square(
        base_height - self._config.reward_config.base_height_target
    )

  # Energy related rewards.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    """Penalizes high torques."""
    return jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    """Penalizes high energy consumption."""
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    """Penalizes jerky actions."""
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    """Penalizes high joint accelerations."""
    return jp.sum(jp.square(qacc))

  # Other rewards.

  def _cost_stand_still(
      self, commands: jax.Array, qpos: jax.Array
  ) -> jax.Array:
    """Penalizes moving while command is zero."""
    # This task has a non-zero command, so this cost should be zero.
    cmd_norm = jp.linalg.norm(commands)
    cost = jp.sum(jp.abs(qpos - self._default_pose))
    cost *= cmd_norm < 0.01
    return cost

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    """Penalizes early termination."""
    return done

  def _reward_alive(self) -> jax.Array:
    """Rewards being alive."""
    return jp.array(1.0)

  # Feet related rewards.

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    """Penalizes feet slipping on the ground."""
    del info  # Unused.
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr] # (4, 3)
    vel_xy = feet_vel[..., :2]
    slip_vel = jp.linalg.norm(vel_xy, axis=-1)
    cost = jp.sum(slip_vel * contact)
    return cost

  def _cost_feet_clearance(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    """Penalizes low feet clearance during swing."""
    del info  # Unused.
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      info: dict[str, Any],
  ) -> jax.Array:
    """Penalizes swing height not reaching target."""
    del info  # Unused.
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact)

  def _reward_feet_air_time(
      self,
      air_time: jax.Array,
      first_contact: jax.Array,
      commands: jax.Array,
      threshold_min: float = 0.1, # Shorter air time for climbing
      threshold_max: float = 0.3,
  ) -> jax.Array:
    """Rewards feet being in the air for a target duration."""
    del commands  # Unused.
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    reward = jp.sum(air_time)
    return reward

  def sample_command(self, rng: jax.Array) -> jax.Array:
    """Returns a fixed command for stair climbing."""
    del rng # Unused, command is fixed.
    
    # Return a constant forward velocity (vx, vy, vyaw)
    return jp.array([
        self._config.lin_vel_x[0], 
        self._config.lin_vel_y[0], 
        self._config.ang_vel_yaw[0]
    ])

def main():
    """Instantiates and tests the environment."""
    print("Instantiating Go1 'Stairs' environment...")
    env = Stairs()
    print("Environment created successfully.")
    
    rng = jax.random.PRNGKey(0)
    
    print("Resetting environment...")
    state = env.reset(rng)
    print(f"Initial observation state shape: {state.obs['state'].shape}")
    print(f"Initial command: {state.info['command']}")
    
    # Test a single step
    action = jp.zeros(env.action_size)
    state = env.step(state, action)
    print(f"Step 1 reward: {state.reward}")
    print(f"Step 1 done: {state.done}")

if __name__ == "__main__":
    main()
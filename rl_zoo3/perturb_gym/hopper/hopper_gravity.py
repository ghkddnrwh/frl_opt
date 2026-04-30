# import numpy as np

# from gymnasium import utils
# from gymnasium.envs.mujoco import MujocoEnv
# from gymnasium.spaces import Box

# import gymnasium as gym
# import xml.etree.ElementTree as ET
# import os

# base_env = "hopper"

# def modify_gravity_in_xml(gravity):
#     # 현재 스크립트의 디렉토리 경로 가져오기
#     file_path = os.path.dirname(__file__)

#     # 원래 XML 파일 경로
#     original_xml_path = os.path.join(os.path.dirname(gym.__file__), "envs", "mujoco", "assets", f"{base_env}.xml")
    
#     # 새로운 중력 값을 적용할 XML 파일 경로
#     perturb_xml_path = os.path.join(file_path, "assets", f"{base_env}_gravity_{gravity}.xml")

#     # XML 파일을 파싱하여 트리 구조 가져오기
#     tree = ET.parse(original_xml_path)
#     root = tree.getroot()

#     # 중력 값을 변경하기 위해 <option> 태그 찾기
#     option_tag = root.find(".//option")
#     if option_tag is not None:
#         original_gravity = option_tag.get("gravity")  # 기존 중력 값 확인
#         print(f"Original gravity: {original_gravity}")

#         # 중력 값을 수정 (예: noise 인자로 받은 값 적용)
#         new_gravity = f"0 0 {gravity}"
#         option_tag.set("gravity", new_gravity)

#         # 변경된 XML 파일 저장
#         tree.write(perturb_xml_path)
#         print(f"Modified XML saved at: {perturb_xml_path}")
#     else:
#         print("No <option> tag found in the XML.")


# DEFAULT_CAMERA_CONFIG = {
#     "trackbodyid": 2,
#     "distance": 3.0,
#     "lookat": np.array((0.0, 0.0, 1.15)),
#     "elevation": -20.0,
# }


# class HopperGravityEnv(MujocoEnv, utils.EzPickle):
#     metadata = {
#         "render_modes": [
#             "human",
#             "rgb_array",
#             "depth_array",
#         ],
#         "render_fps": 125,
#     }

#     def __init__(
#         self,
#         forward_reward_weight=1.0,
#         ctrl_cost_weight=1e-3,
#         healthy_reward=1.0,
#         terminate_when_unhealthy=True,
#         healthy_state_range=(-100.0, 100.0),
#         healthy_z_range=(0.7, float("inf")),
#         healthy_angle_range=(-0.2, 0.2),
#         reset_noise_scale=5e-3,
#         exclude_current_positions_from_observation=True,
#         gravity=None,
#         **kwargs,
#     ):
#         utils.EzPickle.__init__(
#             self,
#             forward_reward_weight,
#             ctrl_cost_weight,
#             healthy_reward,
#             terminate_when_unhealthy,
#             healthy_state_range,
#             healthy_z_range,
#             healthy_angle_range,
#             reset_noise_scale,
#             exclude_current_positions_from_observation,
#             **kwargs,
#         )

#         self._forward_reward_weight = forward_reward_weight

#         self._ctrl_cost_weight = ctrl_cost_weight

#         self._healthy_reward = healthy_reward
#         self._terminate_when_unhealthy = terminate_when_unhealthy

#         self._healthy_state_range = healthy_state_range
#         self._healthy_z_range = healthy_z_range
#         self._healthy_angle_range = healthy_angle_range

#         self._reset_noise_scale = reset_noise_scale

#         self._exclude_current_positions_from_observation = (
#             exclude_current_positions_from_observation
#         )

#         if exclude_current_positions_from_observation:
#             observation_space = Box(
#                 low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64
#             )
#         else:
#             observation_space = Box(
#                 low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64
#             )

#         if gravity == None:
#             perturb_xml_path = f"{base_env}.xml"
#         else:
#             file_path = os.path.dirname(__file__)
        
#             modify_gravity_in_xml(gravity)

#             perturb_xml_path = os.path.join(file_path, "assets", f"{base_env}_gravity_" + str(gravity) + ".xml")
            

#         MujocoEnv.__init__(
#             self,
#             perturb_xml_path,
#             4,
#             observation_space=observation_space,
#             default_camera_config=DEFAULT_CAMERA_CONFIG,
#             **kwargs,
#         )

#     @property
#     def healthy_reward(self):
#         return (
#             float(self.is_healthy or self._terminate_when_unhealthy)
#             * self._healthy_reward
#         )

#     def control_cost(self, action):
#         control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
#         return control_cost

#     @property
#     def is_healthy(self):
#         z, angle = self.data.qpos[1:3]
#         state = self.state_vector()[2:]

#         min_state, max_state = self._healthy_state_range
#         min_z, max_z = self._healthy_z_range
#         min_angle, max_angle = self._healthy_angle_range

#         healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
#         healthy_z = min_z < z < max_z
#         healthy_angle = min_angle < angle < max_angle

#         is_healthy = all((healthy_state, healthy_z, healthy_angle))

#         return is_healthy

#     @property
#     def terminated(self):
#         terminated = not self.is_healthy if self._terminate_when_unhealthy else False
#         return terminated

#     def _get_obs(self):
#         position = self.data.qpos.flat.copy()
#         velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

#         if self._exclude_current_positions_from_observation:
#             position = position[1:]

#         observation = np.concatenate((position, velocity)).ravel()
#         return observation

#     def step(self, action):
#         x_position_before = self.data.qpos[0]
#         self.do_simulation(action, self.frame_skip)
#         x_position_after = self.data.qpos[0]
#         x_velocity = (x_position_after - x_position_before) / self.dt

#         ctrl_cost = self.control_cost(action)

#         forward_reward = self._forward_reward_weight * x_velocity
#         healthy_reward = self.healthy_reward

#         rewards = forward_reward + healthy_reward
#         costs = ctrl_cost

#         observation = self._get_obs()
#         reward = rewards - costs
#         terminated = self.terminated
#         info = {
#             "x_position": x_position_after,
#             "x_velocity": x_velocity,
#         }

#         if self.render_mode == "human":
#             self.render()
#         # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
#         return observation, reward, terminated, False, info

#     def reset_model(self):
#         noise_low = -self._reset_noise_scale
#         noise_high = self._reset_noise_scale

#         qpos = self.init_qpos + self.np_random.uniform(
#             low=noise_low, high=noise_high, size=self.model.nq
#         )
#         qvel = self.init_qvel + self.np_random.uniform(
#             low=noise_low, high=noise_high, size=self.model.nv
#         )

#         self.set_state(qpos, qvel)

#         observation = self._get_obs()
#         return observation

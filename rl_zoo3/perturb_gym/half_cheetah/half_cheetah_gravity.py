# __credits__ = ["Rushiv Arora"]

# import numpy as np

# from gymnasium import utils
# from gymnasium.envs.mujoco import MujocoEnv
# from gymnasium.spaces import Box

# import gymnasium as gym
# import xml.etree.ElementTree as ET
# import os

# base_env = "half_cheetah"

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
#     "distance": 4.0,
# }


# class HalfCheetahGravityEnv(MujocoEnv, utils.EzPickle):
#     metadata = {
#         "render_modes": [
#             "human",
#             "rgb_array",
#             "depth_array",
#         ],
#         "render_fps": 20,
#     }

#     def __init__(
#         self,
#         forward_reward_weight=1.0,
#         ctrl_cost_weight=0.1,
#         reset_noise_scale=0.1,
#         exclude_current_positions_from_observation=True,
#         gravity = None,
#         **kwargs,
#     ):
#         utils.EzPickle.__init__(
#             self,
#             forward_reward_weight,
#             ctrl_cost_weight,
#             reset_noise_scale,
#             exclude_current_positions_from_observation,
#             **kwargs,
#         )

#         self._forward_reward_weight = forward_reward_weight

#         self._ctrl_cost_weight = ctrl_cost_weight

#         self._reset_noise_scale = reset_noise_scale

#         self._exclude_current_positions_from_observation = (
#             exclude_current_positions_from_observation
#         )

#         if exclude_current_positions_from_observation:
#             observation_space = Box(
#                 low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64
#             )
#         else:
#             observation_space = Box(
#                 low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
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
#             5,
#             observation_space=observation_space,
#             default_camera_config=DEFAULT_CAMERA_CONFIG,
#             **kwargs,
#         )

#     def control_cost(self, action):
#         control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
#         return control_cost

#     def step(self, action):
#         x_position_before = self.data.qpos[0]
#         self.do_simulation(action, self.frame_skip)
#         x_position_after = self.data.qpos[0]
#         x_velocity = (x_position_after - x_position_before) / self.dt

#         ctrl_cost = self.control_cost(action)

#         forward_reward = self._forward_reward_weight * x_velocity

#         observation = self._get_obs()
#         reward = forward_reward - ctrl_cost
#         terminated = False
#         info = {
#             "x_position": x_position_after,
#             "x_velocity": x_velocity,
#             "reward_run": forward_reward,
#             "reward_ctrl": -ctrl_cost,
#         }

#         if self.render_mode == "human":
#             self.render()
#         # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
#         return observation, reward, terminated, False, info

#     def _get_obs(self):
#         position = self.data.qpos.flat.copy()
#         velocity = self.data.qvel.flat.copy()

#         if self._exclude_current_positions_from_observation:
#             position = position[1:]

#         observation = np.concatenate((position, velocity)).ravel()
#         return observation

#     def reset_model(self):
#         noise_low = -self._reset_noise_scale
#         noise_high = self._reset_noise_scale

#         qpos = self.init_qpos + self.np_random.uniform(
#             low=noise_low, high=noise_high, size=self.model.nq
#         )
#         qvel = (
#             self.init_qvel
#             + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
#         )

#         self.set_state(qpos, qvel)

#         observation = self._get_obs()
#         return observation

# import numpy as np

# from gymnasium import utils
# from gymnasium.envs.mujoco import MujocoEnv
# from gymnasium.spaces import Box

# import gymnasium as gym
# import xml.etree.ElementTree as ET
# import os

# base_env = "ant"

# import xml.etree.ElementTree as ET
# import numpy as np


# def modify_size_in_xml(perturbation):
#     """
#     XML 내 다리의 길이를 비율 p 만큼 조정하는 함수.
    
#     Args:
#         xml_str (str): XML 파일을 문자열로 입력.
#         p (float): 길이 조정 비율 (예: 0.1이면 길이가 1.1배 증가).
    
#     Returns:
#         str: 수정된 XML 문자열.
#     """
#     # 현재 스크립트의 디렉토리 경로 가져오기
#     file_path = os.path.dirname(__file__)

#     # 원래 XML 파일 경로
#     original_xml_path = os.path.join(os.path.dirname(gym.__file__), "envs", "mujoco", "assets", f"{base_env}.xml")
    
#     # 새로운 중력 값을 적용할 XML 파일 경로
#     perturb_xml_path = os.path.join(file_path, "assets", f"{base_env}_size_{perturbation}.xml")

#     # tree = ET.ElementTree(ET.fromstring(xml_str))
#     # root = tree.getroot()
#     tree = ET.parse(original_xml_path)
#     root = tree.getroot()

#     # 길이 변경 비율
#     scale_factor = 1 + perturbation  # 예: p=0.1이면 1.1배
    
#     for geom in root.findall(".//geom[@fromto]"):  # fromto 속성을 가진 geom 찾기
#         fromto_values = list(map(float, geom.attrib["fromto"].split()))
        
#         x1, y1, z1, x2, y2, z2 = fromto_values
#         vec = np.array([x2 - x1, y2 - y1, z2 - z1])  # 방향 벡터
#         new_vec = vec * scale_factor  # 길이 조정
        
#         new_fromto = f"{x1} {y1} {z1} {x1 + new_vec[0]} {y1 + new_vec[1]} {z1 + new_vec[2]}"
#         geom.set("fromto", new_fromto)

#     tree.write(perturb_xml_path)
    
#     # return ET.tostring(root, encoding="unicode")




# DEFAULT_CAMERA_CONFIG = {
#     "distance": 4.0,
# }


# class AntSizeEnv(MujocoEnv, utils.EzPickle):
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
#         xml_file="ant.xml",
#         ctrl_cost_weight=0.5,
#         use_contact_forces=False,
#         contact_cost_weight=5e-4,
#         healthy_reward=1.0,
#         terminate_when_unhealthy=True,
#         healthy_z_range=(0.2, 1.0),
#         contact_force_range=(-1.0, 1.0),
#         reset_noise_scale=0.1,
#         exclude_current_positions_from_observation=True,
#         perturbation = None,
#         **kwargs,
#     ):
#         if perturbation == None:
#             perturb_xml_path = xml_file
#         else:
#             file_path = os.path.dirname(__file__)
        
#             modify_size_in_xml(perturbation)

#             perturb_xml_path = os.path.join(file_path, "assets", f"{base_env}_size_" + str(perturbation) + ".xml")
            
#         utils.EzPickle.__init__(
#             self,
#             perturb_xml_path,
#             ctrl_cost_weight,
#             use_contact_forces,
#             contact_cost_weight,
#             healthy_reward,
#             terminate_when_unhealthy,
#             healthy_z_range,
#             contact_force_range,
#             reset_noise_scale,
#             exclude_current_positions_from_observation,
#             **kwargs,
#         )

#         self._ctrl_cost_weight = ctrl_cost_weight
#         self._contact_cost_weight = contact_cost_weight

#         self._healthy_reward = healthy_reward
#         self._terminate_when_unhealthy = terminate_when_unhealthy
#         self._healthy_z_range = healthy_z_range

#         self._contact_force_range = contact_force_range

#         self._reset_noise_scale = reset_noise_scale

#         self._use_contact_forces = use_contact_forces

#         self._exclude_current_positions_from_observation = (
#             exclude_current_positions_from_observation
#         )

#         obs_shape = 27
#         if not exclude_current_positions_from_observation:
#             obs_shape += 2
#         if use_contact_forces:
#             obs_shape += 84

#         observation_space = Box(
#             low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
#         )

#         MujocoEnv.__init__(
#             self,
#             perturb_xml_path,
#             5,
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
#     def contact_forces(self):
#         raw_contact_forces = self.data.cfrc_ext
#         min_value, max_value = self._contact_force_range
#         contact_forces = np.clip(raw_contact_forces, min_value, max_value)
#         return contact_forces

#     @property
#     def contact_cost(self):
#         contact_cost = self._contact_cost_weight * np.sum(
#             np.square(self.contact_forces)
#         )
#         return contact_cost

#     @property
#     def is_healthy(self):
#         state = self.state_vector()
#         min_z, max_z = self._healthy_z_range
#         is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
#         return is_healthy

#     @property
#     def terminated(self):
#         terminated = not self.is_healthy if self._terminate_when_unhealthy else False
#         return terminated

#     def step(self, action):
#         xy_position_before = self.get_body_com("torso")[:2].copy()
#         self.do_simulation(action, self.frame_skip)
#         xy_position_after = self.get_body_com("torso")[:2].copy()

#         xy_velocity = (xy_position_after - xy_position_before) / self.dt
#         x_velocity, y_velocity = xy_velocity

#         forward_reward = x_velocity
#         healthy_reward = self.healthy_reward

#         rewards = forward_reward + healthy_reward

#         costs = ctrl_cost = self.control_cost(action)

#         terminated = self.terminated
#         observation = self._get_obs()
#         info = {
#             "reward_forward": forward_reward,
#             "reward_ctrl": -ctrl_cost,
#             "reward_survive": healthy_reward,
#             "x_position": xy_position_after[0],
#             "y_position": xy_position_after[1],
#             "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
#             "x_velocity": x_velocity,
#             "y_velocity": y_velocity,
#             "forward_reward": forward_reward,
#         }
#         if self._use_contact_forces:
#             contact_cost = self.contact_cost
#             costs += contact_cost
#             info["reward_ctrl"] = -contact_cost

#         reward = rewards - costs

#         if self.render_mode == "human":
#             self.render()
#         # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
#         return observation, reward, terminated, False, info

#     def _get_obs(self):
#         position = self.data.qpos.flat.copy()
#         velocity = self.data.qvel.flat.copy()

#         if self._exclude_current_positions_from_observation:
#             position = position[2:]

#         if self._use_contact_forces:
#             contact_force = self.contact_forces.flat.copy()
#             return np.concatenate((position, velocity, contact_force))
#         else:
#             return np.concatenate((position, velocity))

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

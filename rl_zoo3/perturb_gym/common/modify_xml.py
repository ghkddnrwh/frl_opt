# import numpy as np
# import gymnasium as gym
# import xml.etree.ElementTree as ET
# import os

# def modify_xml(base_env, mode='gravity', value=0.0):
#     """
#     MuJoCo XML 파일의 중력 또는 크기를 수정합니다.

#     Args:
#         base_env (str): 환경 이름 ("ant", "half_cheetah", "hopper", "humanoid", "walker2d")
#         mode (str): 'gravity' 또는 'size'
#         value (float): 중력 z값 또는 크기 변화 비율 (예: 0.1 → 10% 증가)
#     """
#     # 파일 경로 설정
#     file_path = os.path.dirname(__file__)
#     original_xml_path = os.path.join(os.path.dirname(gym.__file__), "envs", "mujoco", "assets", f"{base_env}.xml")
#     mode_str = f"{mode}_{value}".replace('.', '_')
#     perturb_xml_path = os.path.join(file_path, "assets", base_env, f"{base_env}_{mode_str}.xml")

#     # XML 파싱
#     tree = ET.parse(original_xml_path)
#     root = tree.getroot()

#     # ANT 전용 limb 구성
#     ANT_LIMBS = {
#         "front_left_leg":  {"aux_1_geom", "left_leg_geom", "left_ankle_geom"},
#         "front_right_leg": {"aux_2_geom", "right_leg_geom", "right_ankle_geom"},
#         "back_left_leg":   {"aux_3_geom", "back_leg_geom", "third_ankle_geom"},
#         "back_right_leg":  {"aux_4_geom", "rightback_leg_geom", "fourth_ankle_geom"},
#         "torso" : {"torso_geom"},
#     }
#     ANT_LIMBS["front_leg"] = ANT_LIMBS["front_left_leg"] | ANT_LIMBS["front_right_leg"]
#     ANT_LIMBS["back_leg"]  = ANT_LIMBS["back_left_leg"]  | ANT_LIMBS["back_right_leg"]
#     ANT_LIMBS["size"]      = {"*"}

#     WALKER2D_LIMBS = {
#         "right_thigh" : {"thigh_geom"},
#         "right_leg" : {"leg_geom"},
#         "right_foot" : {"foot_geom"},
#         "left_thigh" : {"thigh_left_geom"},
#         "left_leg" : {"leg_left_geom"},
#         "left_foot" : {"foot_left_geom"},
#         "torso" : {"torso_geom"},
#     }
#     WALKER2D_LIMBS["thigh"] = WALKER2D_LIMBS["right_thigh"] | WALKER2D_LIMBS["left_thigh"]
#     WALKER2D_LIMBS["leg"] = WALKER2D_LIMBS["right_leg"] | WALKER2D_LIMBS["left_leg"]
#     WALKER2D_LIMBS["foot"] = WALKER2D_LIMBS["right_foot"] | WALKER2D_LIMBS["left_foot"]
#     WALKER2D_LIMBS["right"] = WALKER2D_LIMBS["right_thigh"] | WALKER2D_LIMBS["right_leg"] | WALKER2D_LIMBS["right_foot"]
#     WALKER2D_LIMBS["left"] = WALKER2D_LIMBS["left_thigh"] | WALKER2D_LIMBS["left_leg"] | WALKER2D_LIMBS["left_foot"]
#     WALKER2D_LIMBS["size"] = WALKER2D_LIMBS["right"] | WALKER2D_LIMBS["left"]

#     HALF_CHEETAH_LIMBS = {
#         "bthigh" : {"bthigh"}, 
#         "bshin" : {"bshin"}, 
#         "bfoot" : {"bfoot"}, 
#         "fthigh" : {"fthigh"}, 
#         "fshin" : {"fshin"}, 
#         "ffoot" : {"ffoot"}, 
#         "torso" : {"torso"},
#     }
#     HALF_CHEETAH_LIMBS["thigh"] = HALF_CHEETAH_LIMBS["bthigh"] | HALF_CHEETAH_LIMBS["fthigh"]
#     HALF_CHEETAH_LIMBS["shin"] = HALF_CHEETAH_LIMBS["bshin"] | HALF_CHEETAH_LIMBS["fshin"]
#     HALF_CHEETAH_LIMBS["foot"] = HALF_CHEETAH_LIMBS["bfoot"] | HALF_CHEETAH_LIMBS["ffoot"]
#     HALF_CHEETAH_LIMBS["front"] = HALF_CHEETAH_LIMBS["fthigh"] | HALF_CHEETAH_LIMBS["fshin"] | HALF_CHEETAH_LIMBS["ffoot"]
#     HALF_CHEETAH_LIMBS["back"] = HALF_CHEETAH_LIMBS["bthigh"] | HALF_CHEETAH_LIMBS["bshin"] | HALF_CHEETAH_LIMBS["bfoot"]
#     HALF_CHEETAH_LIMBS["size"] = HALF_CHEETAH_LIMBS["front"] | HALF_CHEETAH_LIMBS["back"]

#     HOPPER_LIMBS = {
#         "thigh" : {"thigh_geom"},
#         "leg" : {"leg_geom"},
#         "foot" : {"foot_geom"},
#         "torso" : {"torso_geom"},
#     }
#     HOPPER_LIMBS["thigh_leg"] = HOPPER_LIMBS["thigh"] | HOPPER_LIMBS["leg"]
#     HOPPER_LIMBS["thigh_foot"] = HOPPER_LIMBS["thigh"] | HOPPER_LIMBS["foot"]
#     HOPPER_LIMBS["leg_foot"] = HOPPER_LIMBS["leg"] | HOPPER_LIMBS["foot"]
#     HOPPER_LIMBS["size"] = HOPPER_LIMBS["thigh"] | HOPPER_LIMBS["leg"] | HOPPER_LIMBS["foot"]
#     HOPPER_LIMBS["torso_thigh"] = HOPPER_LIMBS['torso'] | HOPPER_LIMBS['thigh']
#     HOPPER_LIMBS["torso_leg"] = HOPPER_LIMBS['torso'] | HOPPER_LIMBS['leg']
#     HOPPER_LIMBS['torso_foot'] = HOPPER_LIMBS['torso'] | HOPPER_LIMBS['foot']
#     HOPPER_LIMBS['torso_thigh_leg'] = HOPPER_LIMBS['torso_thigh'] | HOPPER_LIMBS['leg']
#     HOPPER_LIMBS['torso_thigh_foot'] = HOPPER_LIMBS['torso_thigh'] | HOPPER_LIMBS['foot']
#     HOPPER_LIMBS['torso_leg_foot'] = HOPPER_LIMBS['torso_leg'] | HOPPER_LIMBS['foot']
#     HOPPER_LIMBS['torso_thigh_leg_foot'] = HOPPER_LIMBS["torso_thigh_leg"] | HOPPER_LIMBS['foot']
    
#     HUMANOID_LIMBS = {
#         "right_thigh" : {"right_thigh1"}, 
#         "right_shin" : {"right_shin1"}, 
#         "left_thigh" : {"left_thigh1"}, 
#         "left_shin" : {"left_shin1"},
#         "right_uarm" : {"right_uarm1"},
#         "right_larm" : {"right_larm"},
#         "left_uarm" : {"left_uarm1"},
#         "left_larm" : {"left_larm"},
#         "torso" : {"torso1"},
#         "uwaist" : {"uwaist"},
#         "lwaist" : {"lwaist"},
#         "butt" : {"butt"},
#     }
#     HUMANOID_LIMBS["thigh"] = HUMANOID_LIMBS["right_thigh"] | HUMANOID_LIMBS["left_thigh"]
#     HUMANOID_LIMBS["shin"] = HUMANOID_LIMBS["right_shin"] | HUMANOID_LIMBS["left_shin"]
#     HUMANOID_LIMBS["right"] = HUMANOID_LIMBS["right_thigh"] | HUMANOID_LIMBS["right_shin"]
#     HUMANOID_LIMBS["left"] = HUMANOID_LIMBS["left_thigh"] | HUMANOID_LIMBS["left_shin"]
#     HUMANOID_LIMBS["size"] = HUMANOID_LIMBS["right"] | HUMANOID_LIMBS["left"]
#     HUMANOID_LIMBS['uarm'] = HUMANOID_LIMBS['left_uarm'] | HUMANOID_LIMBS['right_uarm']
#     HUMANOID_LIMBS['larm'] = HUMANOID_LIMBS['left_larm'] | HUMANOID_LIMBS['right_larm']
#     HUMANOID_LIMBS['right_arm'] = HUMANOID_LIMBS['right_larm'] | HUMANOID_LIMBS['right_uarm']
#     HUMANOID_LIMBS['left_arm'] = HUMANOID_LIMBS['left_larm'] | HUMANOID_LIMBS['left_uarm']
#     HUMANOID_LIMBS['arm'] = HUMANOID_LIMBS['right_arm'] | HUMANOID_LIMBS['left_arm']
    

#     # 다른 env들에 대한 limb 설정
#     ENV_GEOM_SETTINGS = {
#         "ant": {
#             "method": "fromto",
#             "limbs": ANT_LIMBS
#         },
#         "half_cheetah": {
#             "method": "size",
#             "limbs": HALF_CHEETAH_LIMBS
#         },
#         "hopper": {
#             "method": "size",
#             "limbs": HOPPER_LIMBS
#         },
#         "humanoid": {
#             "method": "fromto",
#             "limbs": HUMANOID_LIMBS
#         },
#         "walker2d": {
#             "method": "size",
#             "limbs": WALKER2D_LIMBS
#         }
#     }

#     scale_factor = 1 + value

#     if mode == 'gravity':
#         option_tag = root.find(".//option")
#         if option_tag is not None:
#             original_gravity = option_tag.get("gravity")
#             if original_gravity is not None:
#                 print(f"Original gravity: {original_gravity}")
#                 try:
#                     gx, gy, gz = map(float, original_gravity.strip().split())
#                     gz *= scale_factor  # z축만 스케일링
#                     new_gravity = f"{gx} {gy} {gz}"
#                     option_tag.set("gravity", new_gravity)
#                 except ValueError:
#                     print("⚠️ gravity 값을 파싱하는 데 실패했습니다.")
#             else:
#                 gz = -9.81 * scale_factor
#                 option_tag.set("gravity", f"0 0 {gz}")
#         else:
#             print("⚠️ No <option> tag found in the XML.")

#     ##############################################################################################################
#     ### Torso
#     ##############################################################################################################
#     # 질량 수정 (예: mode = "mass_torso")
#     elif mode == 'mass_torso':
#         print(f"[*] Modifying torso mass by factor {1 + value}")

#         # 환경별 torso geom 이름 정의
#         TORSO_GEOMS = {
#             "hopper": {"torso_geom"},
#             "half_cheetah": {"torso"},
#             "humanoid": {"torso1"},
#             "ant": {"torso_geom"},
#             "walker2d": {"torso_geom"},
#         }

#         if base_env not in TORSO_GEOMS:
#             raise ValueError(f"Torso mass modification not supported for '{base_env}'")

#         target_torso_geoms = TORSO_GEOMS[base_env]
#         scale_factor = 1 + value

#         for geom in root.findall(".//geom"):
#             name = geom.get("name")
#             if name in target_torso_geoms:
#                 def get_effective_density(geom, root, default_density=1000.0):
#                     """
#                     geom: <geom> Element
#                     root: XML root (tree.getroot())
#                     """
#                     # 1. geom에 직접 density가 있는지 확인
#                     density_attr = geom.get("density")
#                     if density_attr is not None:
#                         return float(density_attr)
                    
#                     # 2. <default><geom> density 확인
#                     default_geom = root.find(".//default/geom")
#                     if default_geom is not None:
#                         default_density_attr = default_geom.get("density")
#                         if default_density_attr is not None:
#                             return float(default_density_attr)
                    
#                     # 3. 없으면 기본값
#                     return default_density

#                 # 적용 예시
#                 mass_attr = geom.get("mass")

#                 if mass_attr is not None:
#                     old_mass = float(mass_attr)
#                     geom.set("mass", f"{old_mass * scale_factor:.6f}")
#                 else:
#                     # tree에서 root 받아서 전달
#                     effective_density = get_effective_density(geom, root)
#                     new_density = effective_density * scale_factor
#                     geom.set("density", f"{new_density:.6f}")

#     ##############################################################################################################
#     ### Torso radius only
#     ##############################################################################################################
#     elif mode == "radius_torso":
#         print(f"[*] Modifying torso RADIUS by factor {1 + value} (length/fromto unchanged)")

#         # 환경별 torso geom 이름 정의 (네가 위에서 mass_torso에 쓰던 것과 동일한 기준)
#         TORSO_GEOMS = {
#             "hopper": {"torso_geom"},
#             "half_cheetah": {"torso"},
#             "humanoid": {"torso1"},
#             "ant": {"torso_geom"},
#             "walker2d": {"torso_geom"},
#         }

#         if base_env not in TORSO_GEOMS:
#             raise ValueError(f"Torso radius modification not supported for '{base_env}'")

#         target_torso_geoms = TORSO_GEOMS[base_env]
#         scale_factor = 1 + value

#         def _safe_parse_floats(s: str):
#             return [float(x) for x in s.strip().split()]

#         changed = 0
#         for geom in root.findall(".//geom"):
#             name = geom.get("name")
#             if name not in target_torso_geoms:
#                 continue

#             gtype = geom.get("type", "sphere")  # mujoco default geom type is sphere
#             size_attr = geom.get("size")
#             if not size_attr:
#                 # size가 없으면 radius를 바꿀 수 없음(대부분은 있음)
#                 print(f"[WARN] torso geom '{name}' has no size attribute -> skip")
#                 continue

#             vals = _safe_parse_floats(size_attr)

#             # capsule/cylinder: size = (radius, half-length) in many models
#             # capsule with fromto: size = (radius) usually (길이는 fromto가 책임)
#             if gtype in ("capsule", "cylinder"):
#                 if len(vals) == 1:
#                     # fromto 캡슐 케이스가 대부분
#                     old_r = vals[0]
#                     new_r = old_r * scale_factor
#                     geom.set("size", f"{new_r:.6f}")
#                     changed += 1
#                 elif len(vals) >= 2:
#                     old_r = vals[0]
#                     new_r = old_r * scale_factor
#                     # half-length(두 번째)는 유지, 나머지 값이 혹시 있어도 보존
#                     vals[0] = new_r
#                     geom.set("size", " ".join([f"{v:.6f}" for v in vals]))
#                     changed += 1
#                 else:
#                     print(f"[WARN] unexpected size format for {gtype} '{name}': '{size_attr}' -> skip")

#             elif gtype == "sphere":
#                 # size = (radius)
#                 old_r = vals[0]
#                 new_r = old_r * scale_factor
#                 geom.set("size", f"{new_r:.6f}")
#                 changed += 1

#             else:
#                 # box/ellipsoid 등은 radius 개념이 애매 -> skip
#                 print(f"[INFO] torso geom '{name}' type='{gtype}' not handled for radius-only -> skip")

#         print(f"[✓] radius_torso done. changed geoms: {changed}")
#     ##############################################################################################################
#     ### Friction: scale effective friction of ALL actual geoms consistently
#     ##############################################################################################################
#     elif mode == "friction":
#         print(f"[*] Modifying ALL geom friction by factor {1 + value}")

#         scale_factor = 1 + value

#         if scale_factor < 0:
#             raise ValueError(
#                 f"Invalid friction scale factor: {scale_factor}. "
#                 "value must be greater than or equal to -1.0"
#             )

#         try:
#             import mujoco
#         except ImportError:
#             raise ImportError(
#                 "This friction mode requires the `mujoco` package. "
#                 "Install it or use a Gymnasium MuJoCo environment that depends on it."
#             )

#         # 1. 원본 XML을 MuJoCo로 compile해서 실제 effective friction을 얻음
#         compiled_model = mujoco.MjModel.from_xml_path(original_xml_path)

#         # 2. 실제 worldbody 안의 geom만 가져옴
#         #    <default><geom ...>은 실제 geom이 아니므로 제외해야 함
#         actual_geoms = root.findall(".//worldbody//geom")

#         if len(actual_geoms) != compiled_model.ngeom:
#             raise RuntimeError(
#                 f"Number of XML geoms ({len(actual_geoms)}) does not match "
#                 f"compiled MuJoCo geoms ({compiled_model.ngeom}). "
#                 "Cannot safely assign friction consistently."
#             )

#         changed = 0

#         for geom_id, geom in enumerate(actual_geoms):
#             name = geom.get("name", f"unnamed_geom_{geom_id}")

#             old_friction = compiled_model.geom_friction[geom_id].copy()
#             new_friction = old_friction * scale_factor

#             geom.set(
#                 "friction",
#                 " ".join(f"{x:.8g}" for x in new_friction)
#             )

#             print(
#                 f"    {geom_id:02d} {name}: "
#                 f"{old_friction} -> {new_friction}"
#             )

#             changed += 1

#         print(f"[✓] friction done. changed actual geoms: {changed}")

#     ##############################################################################################################

#     # 크기(길이) 수정
#     else:
#         if base_env not in ENV_GEOM_SETTINGS:
#             raise ValueError(f"Unknown base_env: {base_env}")

#         config_all = ENV_GEOM_SETTINGS[base_env]
#         method = config_all["method"]

#         if mode not in config_all["limbs"]:
#             raise ValueError(f"Invalid mode '{mode}' for environment '{base_env}'")

#         target_limbs = config_all["limbs"][mode]

#         if method == "fromto":
#             for geom in root.findall(".//geom[@fromto]"):
#                 name = geom.get("name")
#                 if "*" in target_limbs or name in target_limbs:
#                     fromto = list(map(float, geom.attrib["fromto"].split()))
#                     x1, y1, z1, x2, y2, z2 = fromto
#                     vec = np.array([x2 - x1, y2 - y1, z2 - z1])
#                     new_vec = vec * scale_factor
#                     new_fromto = f"{x1} {y1} {z1} {x1 + new_vec[0]} {y1 + new_vec[1]} {z1 + new_vec[2]}"
#                     geom.set("fromto", new_fromto)

#         elif method == "size":
#             for geom in root.findall(".//geom"):
#                 name = geom.get("name")
#                 if name in target_limbs:
#                     size_attr = geom.get("size")
#                     if size_attr:
#                         size_values = size_attr.split()
#                         if len(size_values) == 2:
#                             original_size = float(size_values[1])
#                             new_size = original_size * scale_factor
#                             geom.set("size", f"{size_values[0]} {new_size:.6f}")

#     tree.write(perturb_xml_path)
#     print(f"[✓] Modified XML saved at: {perturb_xml_path}")

#     return perturb_xml_path


import numpy as np
import gymnasium as gym
import xml.etree.ElementTree as ET
import os

def modify_xml(base_env, mode='gravity', value=0.0):
    """
    MuJoCo XML 파일의 중력 또는 크기를 수정합니다.

    Args:
        base_env (str): 환경 이름 ("ant", "half_cheetah", "hopper", "humanoid", "walker2d")
        mode (str): 'gravity' 또는 'size'
        value (float): 중력 z값 또는 크기 변화 비율 (예: 0.1 → 10% 증가)
    """
    # 파일 경로 설정
    file_path = os.path.dirname(__file__)
    original_xml_path = os.path.join(os.path.dirname(gym.__file__), "envs", "mujoco", "assets", f"{base_env}.xml")
    mode_str = f"{mode}_{value}".replace('.', '_')
    perturb_xml_path = os.path.join(file_path, "assets", base_env, f"{base_env}_{mode_str}.xml")

    # XML 파싱
    tree = ET.parse(original_xml_path)
    root = tree.getroot()

    # ANT 전용 limb 구성
    ANT_LIMBS = {
        "front_left_leg":  {"aux_1_geom", "left_leg_geom", "left_ankle_geom"},
        "front_right_leg": {"aux_2_geom", "right_leg_geom", "right_ankle_geom"},
        "back_left_leg":   {"aux_3_geom", "back_leg_geom", "third_ankle_geom"},
        "back_right_leg":  {"aux_4_geom", "rightback_leg_geom", "fourth_ankle_geom"},
        "torso" : {"torso_geom"},
    }
    ANT_LIMBS["front_leg"] = ANT_LIMBS["front_left_leg"] | ANT_LIMBS["front_right_leg"]
    ANT_LIMBS["back_leg"]  = ANT_LIMBS["back_left_leg"]  | ANT_LIMBS["back_right_leg"]
    ANT_LIMBS["size"]      = {"*"}

    WALKER2D_LIMBS = {
        "right_thigh" : {"thigh_geom"},
        "right_leg" : {"leg_geom"},
        "right_foot" : {"foot_geom"},
        "left_thigh" : {"thigh_left_geom"},
        "left_leg" : {"leg_left_geom"},
        "left_foot" : {"foot_left_geom"},
        "torso" : {"torso_geom"},
    }
    WALKER2D_LIMBS["thigh"] = WALKER2D_LIMBS["right_thigh"] | WALKER2D_LIMBS["left_thigh"]
    WALKER2D_LIMBS["leg"] = WALKER2D_LIMBS["right_leg"] | WALKER2D_LIMBS["left_leg"]
    WALKER2D_LIMBS["foot"] = WALKER2D_LIMBS["right_foot"] | WALKER2D_LIMBS["left_foot"]
    WALKER2D_LIMBS["right"] = WALKER2D_LIMBS["right_thigh"] | WALKER2D_LIMBS["right_leg"] | WALKER2D_LIMBS["right_foot"]
    WALKER2D_LIMBS["left"] = WALKER2D_LIMBS["left_thigh"] | WALKER2D_LIMBS["left_leg"] | WALKER2D_LIMBS["left_foot"]
    WALKER2D_LIMBS["size"] = WALKER2D_LIMBS["right"] | WALKER2D_LIMBS["left"]

    HALF_CHEETAH_LIMBS = {
        "bthigh" : {"bthigh"}, 
        "bshin" : {"bshin"}, 
        "bfoot" : {"bfoot"}, 
        "fthigh" : {"fthigh"}, 
        "fshin" : {"fshin"}, 
        "ffoot" : {"ffoot"}, 
        "torso" : {"torso"},
    }
    HALF_CHEETAH_LIMBS["thigh"] = HALF_CHEETAH_LIMBS["bthigh"] | HALF_CHEETAH_LIMBS["fthigh"]
    HALF_CHEETAH_LIMBS["shin"] = HALF_CHEETAH_LIMBS["bshin"] | HALF_CHEETAH_LIMBS["fshin"]
    HALF_CHEETAH_LIMBS["foot"] = HALF_CHEETAH_LIMBS["bfoot"] | HALF_CHEETAH_LIMBS["ffoot"]
    HALF_CHEETAH_LIMBS["front"] = HALF_CHEETAH_LIMBS["fthigh"] | HALF_CHEETAH_LIMBS["fshin"] | HALF_CHEETAH_LIMBS["ffoot"]
    HALF_CHEETAH_LIMBS["back"] = HALF_CHEETAH_LIMBS["bthigh"] | HALF_CHEETAH_LIMBS["bshin"] | HALF_CHEETAH_LIMBS["bfoot"]
    HALF_CHEETAH_LIMBS["size"] = HALF_CHEETAH_LIMBS["front"] | HALF_CHEETAH_LIMBS["back"]

    HOPPER_LIMBS = {
        "thigh" : {"thigh_geom"},
        "leg" : {"leg_geom"},
        "foot" : {"foot_geom"},
        "torso" : {"torso_geom"},
    }
    HOPPER_LIMBS["thigh_leg"] = HOPPER_LIMBS["thigh"] | HOPPER_LIMBS["leg"]
    HOPPER_LIMBS["thigh_foot"] = HOPPER_LIMBS["thigh"] | HOPPER_LIMBS["foot"]
    HOPPER_LIMBS["leg_foot"] = HOPPER_LIMBS["leg"] | HOPPER_LIMBS["foot"]
    HOPPER_LIMBS["size"] = HOPPER_LIMBS["thigh"] | HOPPER_LIMBS["leg"] | HOPPER_LIMBS["foot"]
    HOPPER_LIMBS["torso_thigh"] = HOPPER_LIMBS['torso'] | HOPPER_LIMBS['thigh']
    HOPPER_LIMBS["torso_leg"] = HOPPER_LIMBS['torso'] | HOPPER_LIMBS['leg']
    HOPPER_LIMBS['torso_foot'] = HOPPER_LIMBS['torso'] | HOPPER_LIMBS['foot']
    HOPPER_LIMBS['torso_thigh_leg'] = HOPPER_LIMBS['torso_thigh'] | HOPPER_LIMBS['leg']
    HOPPER_LIMBS['torso_thigh_foot'] = HOPPER_LIMBS['torso_thigh'] | HOPPER_LIMBS['foot']
    HOPPER_LIMBS['torso_leg_foot'] = HOPPER_LIMBS['torso_leg'] | HOPPER_LIMBS['foot']
    HOPPER_LIMBS['torso_thigh_leg_foot'] = HOPPER_LIMBS["torso_thigh_leg"] | HOPPER_LIMBS['foot']
    
    HUMANOID_LIMBS = {
        "right_thigh" : {"right_thigh1"}, 
        "right_shin" : {"right_shin1"}, 
        "left_thigh" : {"left_thigh1"}, 
        "left_shin" : {"left_shin1"},
        "right_uarm" : {"right_uarm1"},
        "right_larm" : {"right_larm"},
        "left_uarm" : {"left_uarm1"},
        "left_larm" : {"left_larm"},
        "torso" : {"torso1"},
        "uwaist" : {"uwaist"},
        "lwaist" : {"lwaist"},
        "butt" : {"butt"},
    }
    HUMANOID_LIMBS["thigh"] = HUMANOID_LIMBS["right_thigh"] | HUMANOID_LIMBS["left_thigh"]
    HUMANOID_LIMBS["shin"] = HUMANOID_LIMBS["right_shin"] | HUMANOID_LIMBS["left_shin"]
    HUMANOID_LIMBS["right"] = HUMANOID_LIMBS["right_thigh"] | HUMANOID_LIMBS["right_shin"]
    HUMANOID_LIMBS["left"] = HUMANOID_LIMBS["left_thigh"] | HUMANOID_LIMBS["left_shin"]
    HUMANOID_LIMBS["size"] = HUMANOID_LIMBS["right"] | HUMANOID_LIMBS["left"]
    HUMANOID_LIMBS['uarm'] = HUMANOID_LIMBS['left_uarm'] | HUMANOID_LIMBS['right_uarm']
    HUMANOID_LIMBS['larm'] = HUMANOID_LIMBS['left_larm'] | HUMANOID_LIMBS['right_larm']
    HUMANOID_LIMBS['right_arm'] = HUMANOID_LIMBS['right_larm'] | HUMANOID_LIMBS['right_uarm']
    HUMANOID_LIMBS['left_arm'] = HUMANOID_LIMBS['left_larm'] | HUMANOID_LIMBS['left_uarm']
    HUMANOID_LIMBS['arm'] = HUMANOID_LIMBS['right_arm'] | HUMANOID_LIMBS['left_arm']
    

    # 다른 env들에 대한 limb 설정
    ENV_GEOM_SETTINGS = {
        "ant": {
            "method": "fromto",
            "limbs": ANT_LIMBS
        },
        "half_cheetah": {
            "method": "size",
            "limbs": HALF_CHEETAH_LIMBS
        },
        "hopper": {
            "method": "size",
            "limbs": HOPPER_LIMBS
        },
        "humanoid": {
            "method": "fromto",
            "limbs": HUMANOID_LIMBS
        },
        "walker2d": {
            "method": "size",
            "limbs": WALKER2D_LIMBS
        }
    }

    scale_factor = 1 + value

    if mode == 'gravity':
        option_tag = root.find(".//option")
        if option_tag is not None:
            original_gravity = option_tag.get("gravity")
            if original_gravity is not None:
                print(f"Original gravity: {original_gravity}")
                try:
                    gx, gy, gz = map(float, original_gravity.strip().split())
                    gz *= scale_factor  # z축만 스케일링
                    new_gravity = f"{gx} {gy} {gz}"
                    option_tag.set("gravity", new_gravity)
                except ValueError:
                    print("⚠️ gravity 값을 파싱하는 데 실패했습니다.")
            else:
                gz = -9.81 * scale_factor
                option_tag.set("gravity", f"0 0 {gz}")
        else:
            print("⚠️ No <option> tag found in the XML.")

    ##############################################################################################################
    ### Torso
    ##############################################################################################################
    # 질량 수정 (예: mode = "mass_torso")
    elif mode == 'mass_torso':
        print(f"[*] Modifying torso mass by factor {1 + value}")

        # 환경별 torso geom 이름 정의
        TORSO_GEOMS = {
            "hopper": {"torso_geom"},
            "half_cheetah": {"torso"},
            "humanoid": {"torso1"},
            "ant": {"torso_geom"},
            "walker2d": {"torso_geom"},
        }

        if base_env not in TORSO_GEOMS:
            raise ValueError(f"Torso mass modification not supported for '{base_env}'")

        target_torso_geoms = TORSO_GEOMS[base_env]
        scale_factor = 1 + value

        for geom in root.findall(".//geom"):
            name = geom.get("name")
            if name in target_torso_geoms:
                def get_effective_density(geom, root, default_density=1000.0):
                    """
                    geom: <geom> Element
                    root: XML root (tree.getroot())
                    """
                    # 1. geom에 직접 density가 있는지 확인
                    density_attr = geom.get("density")
                    if density_attr is not None:
                        return float(density_attr)
                    
                    # 2. <default><geom> density 확인
                    default_geom = root.find(".//default/geom")
                    if default_geom is not None:
                        default_density_attr = default_geom.get("density")
                        if default_density_attr is not None:
                            return float(default_density_attr)
                    
                    # 3. 없으면 기본값
                    return default_density

                # 적용 예시
                mass_attr = geom.get("mass")

                if mass_attr is not None:
                    old_mass = float(mass_attr)
                    geom.set("mass", f"{old_mass * scale_factor:.6f}")
                else:
                    # tree에서 root 받아서 전달
                    effective_density = get_effective_density(geom, root)
                    new_density = effective_density * scale_factor
                    geom.set("density", f"{new_density:.6f}")

    ##############################################################################################################
    ### Torso radius only
    ##############################################################################################################
    elif mode == "radius_torso":
        print(f"[*] Modifying torso RADIUS by factor {1 + value} (length/fromto unchanged)")

        # 환경별 torso geom 이름 정의 (네가 위에서 mass_torso에 쓰던 것과 동일한 기준)
        TORSO_GEOMS = {
            "hopper": {"torso_geom"},
            "half_cheetah": {"torso"},
            "humanoid": {"torso1"},
            "ant": {"torso_geom"},
            "walker2d": {"torso_geom"},
        }

        if base_env not in TORSO_GEOMS:
            raise ValueError(f"Torso radius modification not supported for '{base_env}'")

        target_torso_geoms = TORSO_GEOMS[base_env]
        scale_factor = 1 + value

        def _safe_parse_floats(s: str):
            return [float(x) for x in s.strip().split()]

        changed = 0
        for geom in root.findall(".//geom"):
            name = geom.get("name")
            if name not in target_torso_geoms:
                continue

            gtype = geom.get("type", "sphere")  # mujoco default geom type is sphere
            size_attr = geom.get("size")
            if not size_attr:
                # size가 없으면 radius를 바꿀 수 없음(대부분은 있음)
                print(f"[WARN] torso geom '{name}' has no size attribute -> skip")
                continue

            vals = _safe_parse_floats(size_attr)

            # capsule/cylinder: size = (radius, half-length) in many models
            # capsule with fromto: size = (radius) usually (길이는 fromto가 책임)
            if gtype in ("capsule", "cylinder"):
                if len(vals) == 1:
                    # fromto 캡슐 케이스가 대부분
                    old_r = vals[0]
                    new_r = old_r * scale_factor
                    geom.set("size", f"{new_r:.6f}")
                    changed += 1
                elif len(vals) >= 2:
                    old_r = vals[0]
                    new_r = old_r * scale_factor
                    # half-length(두 번째)는 유지, 나머지 값이 혹시 있어도 보존
                    vals[0] = new_r
                    geom.set("size", " ".join([f"{v:.6f}" for v in vals]))
                    changed += 1
                else:
                    print(f"[WARN] unexpected size format for {gtype} '{name}': '{size_attr}' -> skip")

            elif gtype == "sphere":
                # size = (radius)
                old_r = vals[0]
                new_r = old_r * scale_factor
                geom.set("size", f"{new_r:.6f}")
                changed += 1

            else:
                # box/ellipsoid 등은 radius 개념이 애매 -> skip
                print(f"[INFO] torso geom '{name}' type='{gtype}' not handled for radius-only -> skip")

        print(f"[✓] radius_torso done. changed geoms: {changed}")
    ##############################################################################################################
    ### Friction: scale effective friction of ALL actual geoms consistently
    ##############################################################################################################
    elif mode == "friction":
        print(f"[*] Modifying ALL geom friction by factor {1 + value}")

        scale_factor = 1 + value

        if scale_factor < 0:
            raise ValueError(
                f"Invalid friction scale factor: {scale_factor}. "
                "value must be greater than or equal to -1.0"
            )

        try:
            import mujoco
        except ImportError:
            raise ImportError(
                "This friction mode requires the `mujoco` package. "
                "Install it or use a Gymnasium MuJoCo environment that depends on it."
            )

        # 1. 원본 XML을 MuJoCo로 compile해서 실제 effective friction을 얻음
        compiled_model = mujoco.MjModel.from_xml_path(original_xml_path)

        # 2. 실제 worldbody 안의 geom만 가져옴
        #    <default><geom ...>은 실제 geom이 아니므로 제외해야 함
        actual_geoms = root.findall(".//worldbody//geom")

        if len(actual_geoms) != compiled_model.ngeom:
            raise RuntimeError(
                f"Number of XML geoms ({len(actual_geoms)}) does not match "
                f"compiled MuJoCo geoms ({compiled_model.ngeom}). "
                "Cannot safely assign friction consistently."
            )

        changed = 0

        for geom_id, geom in enumerate(actual_geoms):
            name = geom.get("name", f"unnamed_geom_{geom_id}")

            old_friction = compiled_model.geom_friction[geom_id].copy()
            new_friction = old_friction * scale_factor

            geom.set(
                "friction",
                " ".join(f"{x:.8g}" for x in new_friction)
            )

            print(
                f"    {geom_id:02d} {name}: "
                f"{old_friction} -> {new_friction}"
            )

            changed += 1

        print(f"[✓] friction done. changed actual geoms: {changed}")


    ##############################################################################################################
    ### Additional COMMON perturbations for all 5 MuJoCo envs
    ### - Existing perturbation modes above, including friction, are intentionally untouched.
    ### - New modes use the prefix "all_" to avoid conflicts with existing limb-specific modes.
    ### - Each mode writes an explicit value to every target element so default-inherited values are not skipped.
    ##############################################################################################################
    elif mode in {
        "all_damping",
        "all_armature",
        "all_stiffness",
        "all_gear",
        "all_ctrlrange",
        "all_geom_mass",
        "all_geom_density",
        "all_margin",
    }:
        print(f"[*] Modifying COMMON perturbation '{mode}' by factor {1 + value}")

        scale_factor = 1 + value
        if scale_factor < 0:
            raise ValueError(
                f"Invalid scale factor: {scale_factor}. "
                "value must be greater than or equal to -1.0"
            )

        def _parse_floats(s):
            return [float(x) for x in s.strip().split()]

        def _fmt(values):
            return " ".join(f"{v:.8g}" for v in values)

        def _first_default_attr(tag_name, attr_name):
            """
            Read the first matching default attribute, e.g. <default><joint damping="..."/>.
            This keeps the behavior simple and consistent with the original code style.
            """
            default_elem = root.find(f".//default/{tag_name}")
            if default_elem is None:
                return None
            return default_elem.get(attr_name)

        def _effective_attr(elem, tag_name, attr_name, fallback=None):
            """
            Get direct attribute first; if absent, use the default tag attribute;
            if still absent, use a MuJoCo-style fallback supplied by the caller.
            """
            direct = elem.get(attr_name)
            if direct is not None:
                return direct
            inherited = _first_default_attr(tag_name, attr_name)
            if inherited is not None:
                return inherited
            return fallback

        def _scale_scalar_for_all(elements, tag_name, attr_name, fallback):
            expected = len(elements)
            changed = 0

            for idx, elem in enumerate(elements):
                old_attr = _effective_attr(elem, tag_name, attr_name, fallback=fallback)
                if old_attr is None:
                    raise RuntimeError(
                        f"Cannot determine effective {attr_name} for {tag_name}[{idx}]."
                    )

                old_value = float(old_attr)
                new_value = old_value * scale_factor
                elem.set(attr_name, f"{new_value:.8g}")
                changed += 1

            if changed != expected:
                raise RuntimeError(
                    f"Mode '{mode}' changed {changed}/{expected} <{tag_name}> elements. "
                    "This should never happen for an all-elements perturbation."
                )

            print(f"[✓] {mode} done. changed {tag_name} elements: {changed}")

        def _scale_vector_for_all(elements, tag_name, attr_name, fallback, center_preserving=False):
            expected = len(elements)
            changed = 0

            for idx, elem in enumerate(elements):
                old_attr = _effective_attr(elem, tag_name, attr_name, fallback=fallback)
                if old_attr is None:
                    raise RuntimeError(
                        f"Cannot determine effective {attr_name} for {tag_name}[{idx}]."
                    )

                old_values = _parse_floats(old_attr)

                if center_preserving:
                    if len(old_values) != 2:
                        raise RuntimeError(
                            f"center_preserving=True requires 2 values for {attr_name}, "
                            f"but got {old_attr} for {tag_name}[{idx}]."
                        )
                    lo, hi = old_values
                    center = 0.5 * (lo + hi)
                    half_width = 0.5 * (hi - lo) * scale_factor
                    new_values = [center - half_width, center + half_width]
                else:
                    new_values = [x * scale_factor for x in old_values]

                elem.set(attr_name, _fmt(new_values))
                changed += 1

            if changed != expected:
                raise RuntimeError(
                    f"Mode '{mode}' changed {changed}/{expected} <{tag_name}> elements. "
                    "This should never happen for an all-elements perturbation."
                )

            print(f"[✓] {mode} done. changed {tag_name} elements: {changed}")

        actual_geoms = root.findall(".//worldbody//geom")
        actual_joints = root.findall(".//worldbody//joint")
        actual_motors = root.findall(".//actuator//motor")

        if mode == "all_damping":
            # Apply to every actual joint, including root/free/slide joints.
            # Explicit zeros remain zero after multiplicative scaling.
            _scale_scalar_for_all(
                actual_joints,
                tag_name="joint",
                attr_name="damping",
                fallback="0.0",
            )

        elif mode == "all_armature":
            _scale_scalar_for_all(
                actual_joints,
                tag_name="joint",
                attr_name="armature",
                fallback="0.0",
            )

        elif mode == "all_stiffness":
            _scale_scalar_for_all(
                actual_joints,
                tag_name="joint",
                attr_name="stiffness",
                fallback="0.0",
            )

        elif mode == "all_gear":
            # gear can be a scalar or a vector. Scale every component for every motor.
            _scale_vector_for_all(
                actual_motors,
                tag_name="motor",
                attr_name="gear",
                fallback="1.0",
                center_preserving=False,
            )

        elif mode == "all_ctrlrange":
            # Preserve the center of the control range and scale only the width.
            # This avoids shifting the neutral action.
            _scale_vector_for_all(
                actual_motors,
                tag_name="motor",
                attr_name="ctrlrange",
                fallback="-1.0 1.0",
                center_preserving=True,
            )

        elif mode in {"all_geom_mass", "all_geom_density"}:
            # Apply to every actual geom in worldbody. If a geom has explicit mass, scale mass.
            # Otherwise, write an explicit density obtained from the direct/default/fallback value.
            # For static geoms such as floor planes, MuJoCo may ignore mass/density dynamically,
            # but we still write the attribute so the perturbation is not silently skipped.
            expected = len(actual_geoms)
            changed = 0

            for idx, geom in enumerate(actual_geoms):
                mass_attr = geom.get("mass")

                if mass_attr is not None:
                    # If explicit mass exists, MuJoCo uses it instead of density.
                    # Scale mass for both aliases so the perturbation is physically applied.
                    old_mass = float(mass_attr)
                    geom.set("mass", f"{old_mass * scale_factor:.8g}")
                else:
                    density_attr = _effective_attr(
                        geom,
                        tag_name="geom",
                        attr_name="density",
                        fallback="1000.0",
                    )
                    if density_attr is None:
                        raise RuntimeError(
                            f"Cannot determine effective density for geom[{idx}]."
                        )
                    old_density = float(density_attr)
                    geom.set("density", f"{old_density * scale_factor:.8g}")

                changed += 1

            if changed != expected:
                raise RuntimeError(
                    f"Mode '{mode}' changed {changed}/{expected} geoms. "
                    "This should never happen for an all-elements perturbation."
                )

            print(f"[✓] {mode} done. changed actual geoms: {changed}")

        elif mode == "all_margin":
            _scale_scalar_for_all(
                actual_geoms,
                tag_name="geom",
                attr_name="margin",
                fallback="0.0",
            )

    ##############################################################################################################

    # 크기(길이) 수정
    else:
        if base_env not in ENV_GEOM_SETTINGS:
            raise ValueError(f"Unknown base_env: {base_env}")

        config_all = ENV_GEOM_SETTINGS[base_env]
        method = config_all["method"]

        if mode not in config_all["limbs"]:
            raise ValueError(f"Invalid mode '{mode}' for environment '{base_env}'")

        target_limbs = config_all["limbs"][mode]

        if method == "fromto":
            for geom in root.findall(".//geom[@fromto]"):
                name = geom.get("name")
                if "*" in target_limbs or name in target_limbs:
                    fromto = list(map(float, geom.attrib["fromto"].split()))
                    x1, y1, z1, x2, y2, z2 = fromto
                    vec = np.array([x2 - x1, y2 - y1, z2 - z1])
                    new_vec = vec * scale_factor
                    new_fromto = f"{x1} {y1} {z1} {x1 + new_vec[0]} {y1 + new_vec[1]} {z1 + new_vec[2]}"
                    geom.set("fromto", new_fromto)

        elif method == "size":
            for geom in root.findall(".//geom"):
                name = geom.get("name")
                if name in target_limbs:
                    size_attr = geom.get("size")
                    if size_attr:
                        size_values = size_attr.split()
                        if len(size_values) == 2:
                            original_size = float(size_values[1])
                            new_size = original_size * scale_factor
                            geom.set("size", f"{size_values[0]} {new_size:.6f}")

    tree.write(perturb_xml_path)
    print(f"[✓] Modified XML saved at: {perturb_xml_path}")

    return perturb_xml_path
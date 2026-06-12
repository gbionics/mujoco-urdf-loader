import tempfile
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

from mujoco_urdf_loader.generator import load_urdf_into_mjcf
from mujoco_urdf_loader.hands_fcn import (
    add_hand_actuators,
    add_hand_equalities,
    add_wrist_actuators,
    set_thumb_angle,
)
from mujoco_urdf_loader.mjcf_fcn import (
    add_new_worldbody,
    separate_left_right_collision_groups,
    set_joint_damping,
)
from mujoco_urdf_loader.urdf_fcn import (
    add_mujoco_element,
    get_mesh_path,
    get_robot_urdf,
    resolve_mesh_filenames,
    remove_gazebo_elements,
    remove_links_and_joints_by_keep_list,
)

# Load the robot urdf
robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")

# find the mesh path
mesh_path = get_mesh_path(robot_urdf)

# remove the gazebo elements
robot_urdf = remove_gazebo_elements(robot_urdf)

# MuJoCo cannot resolve package:// mesh filenames directly.
robot_urdf = resolve_mesh_filenames(robot_urdf)

# add the mujoco element to be able to load the urdf into the mjcf
robot_urdf = add_mujoco_element(robot_urdf, mesh_path)

to_keep = [
    "r_hand",
    "r_wrist",
    "r_forearm",
    "r_pinkie",
    "r_ring",
    "r_middle",
    "r_index",
    "r_thumb",
]

robot_urdf_hand = remove_links_and_joints_by_keep_list(robot_urdf, to_keep)

mjcf_hand = load_urdf_into_mjcf(robot_urdf_hand)

add_new_worldbody(mjcf_hand, freeze_root=True, r_x=np.pi / 2)

add_hand_equalities(mjcf_hand)

set_thumb_angle(mjcf_hand, angle=9 * 5)

hand_elements = ["thumb", "index", "middle", "ring", "pinkie"]

add_hand_actuators(mjcf_hand, hand_elements=hand_elements)

add_wrist_actuators(mjcf_hand)

set_joint_damping(mjcf_hand, subset=hand_elements, damping=0.005)

separate_left_right_collision_groups(mjcf_hand)

hand_str = ET.tostring(mjcf_hand, encoding="unicode", method="xml")

# save the model
with open("ergoCub_hand.xml", "w") as f:
    f.write(hand_str)

# save the model to a temporary file
with tempfile.NamedTemporaryFile(mode="w+") as path_temp_xml:
    path_temp_xml.write(hand_str)
    world_str = f"""
    <mujoco model="ergoCubWorld">
        <include file="{path_temp_xml.name}"/>

        <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20"/>
        </visual>

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
            <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
            markrgb="0.8 0.8 0.8" width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
        </asset>

        <worldbody>
            <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
            <camera name="default" pos="0.846 -1.465 0.916" xyaxes="0.866 0.500 0.000 -0.171 0.296 0.940"/>
            <geom name="floor" pos="0 0 -0.1" size="0 0 0.05" type="plane" material="groundplane"/>
            <body name="box01" pos="0 0.25 0.05">
                <freejoint/>
                <geom name="box01_geom" size="0.025 0.025 0.025" type="box" rgba="1 0 0 1" mass="0.2"/>
            </body>
        </worldbody>
    </mujoco>
    """

    # Load the model in mujoco and visualize it
    model = mujoco.MjModel.from_xml_string(world_str)
    data = mujoco.MjData(model)

    # Visualize the model
    mujoco.viewer.launch(model=model, data=data)

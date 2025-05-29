import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import resolve_robotics_uri_py as rru

from mujoco_urdf_loader import (
    ControlMode,
    MujocoWrapper,
    URDFtoMuJoCoLoader,
    URDFtoMuJoCoLoaderCfg,
)
from mujoco_urdf_loader.urdf_fcn import get_mesh_path

controlled_joints = [
    "l_hip_pitch",
    "r_hip_pitch",
    "torso_roll",
    "l_hip_roll",
    "r_hip_roll",
    "torso_pitch",
    "torso_yaw",
    "l_hip_yaw",
    "r_hip_yaw",
    "l_shoulder_pitch",
    "neck_pitch",
    "r_shoulder_pitch",
    "l_knee",
    "r_knee",
    "l_shoulder_roll",
    "neck_roll",
    "r_shoulder_roll",
    "l_ankle_pitch",
    "r_ankle_pitch",
    "neck_yaw",
    "l_ankle_roll",
    "r_ankle_roll",
    "l_shoulder_yaw",
    "r_shoulder_yaw",
    "l_elbow",
    "r_elbow",
]

control_modes = [ControlMode.TORQUE] * len(controlled_joints)
stiffness = [0.0] * len(controlled_joints)
damping = [0.0] * len(controlled_joints)

cfg = URDFtoMuJoCoLoaderCfg(controlled_joints, control_modes, stiffness, damping)

urdf_string = str(
    rru.resolve_robotics_uri("package://ergoCub/robots/ergoCubSN002/model.urdf")
)
mesh_path = get_mesh_path(ET.parse(urdf_string).getroot())

loader = URDFtoMuJoCoLoader.load_urdf(urdf_string, mesh_path, cfg)

# save xml_str to a file
path = "ergocub.xml"
with open(path, "w") as f:
    f.write(loader.get_mjcf_string())

        # include the model in a simple world
world_str = f"""
    <mujoco model="ergoCubWorld">
        <include file="{path}"/>

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
            <geom name="floor" pos="0 0 0" size="0 0 0.05" type="plane" material="groundplane"/>
        </worldbody>
    </mujoco>
    """

mujoco_wrapper = MujocoWrapper(world_str)
model = mujoco_wrapper.model
data = mujoco_wrapper.data

# or directly load the model in mujoco

model = mujoco.MjModel.from_xml_string(world_str)
data = mujoco.MjData(model)

data.qpos[2] = 1.5
# visualize the model
mujoco.viewer.launch(model, data)

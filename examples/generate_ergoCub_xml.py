from mujoco_urdf_loader import MujocoWrapper
from mujoco_urdf_loader import URDFtoMuJoCoLoader, URDFtoMuJoCoLoaderCfg, ControlMode
import resolve_robotics_uri_py as rru
import mujoco

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

loader = URDFtoMuJoCoLoader.load_urdf(urdf_string, cfg)

# save xml_str to a file
path = "ergocub.xml"
with open(path, "w") as f:
    f.write(loader.get_mjcf_string())

mujoco_wrapper = MujocoWrapper(loader.get_mjcf_string())
model = mujoco_wrapper.model
data = mujoco_wrapper.data

# or directly load the model in mujoco

model = mujoco.MjModel.from_xml_string(loader.get_mjcf_string())
data = mujoco.MjData(model)


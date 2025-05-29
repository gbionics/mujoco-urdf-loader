from xml.etree import ElementTree as ET

import idyntree.bindings as idyntree
import mujoco
import pytest
import resolve_robotics_uri_py as rru

# Import the ANYmal robot description
from robot_descriptions import anymal_c_description as anymal_description

from mujoco_urdf_loader import ControlMode, URDFtoMuJoCoLoader, URDFtoMuJoCoLoaderCfg
from mujoco_urdf_loader.urdf_fcn import get_mesh_path

# Define robot configurations in a dictionary
ROBOTS = {
    "ergoCub": {
        "controlled_joints": [
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
        ],
        "urdf_path": str(
            rru.resolve_robotics_uri("package://ergoCub/robots/ergoCubSN002/model.urdf")
        ),
        "mesh_path": None,  # Will be set later
    },
    "ANYmal": {
        "controlled_joints": [
            "LF_HAA",
            "LF_HFE",
            "LF_KFE",
            "RF_HAA",
            "RF_HFE",
            "RF_KFE",
            "LH_HAA",
            "LH_HFE",
            "LH_KFE",
            "RH_HAA",
            "RH_HFE",
            "RH_KFE",
        ],
        "urdf_path": anymal_description.URDF_PATH,
        "mesh_path": f"{anymal_description.PACKAGE_PATH}/meshes",
    },
}

# Update mesh paths for robots that need it
for robot in ROBOTS.values():
    if robot["mesh_path"] is None:
        robot["mesh_path"] = get_mesh_path(ET.parse(robot["urdf_path"]).getroot())


@pytest.mark.parametrize("robot_name", ["ergoCub", "ANYmal"])
def test_load_urdf_into_mjcf(robot_name):
    robot = ROBOTS[robot_name]
    controlled_joints = robot["controlled_joints"]
    urdf_path = robot["urdf_path"]
    mesh_path = robot["mesh_path"]

    control_modes = [ControlMode.TORQUE] * len(controlled_joints)
    stiffness = [0.0] * len(controlled_joints)
    damping = [0.0] * len(controlled_joints)
    cfg = URDFtoMuJoCoLoaderCfg(controlled_joints, control_modes, stiffness, damping)

    loader = URDFtoMuJoCoLoader.load_urdf(urdf_path, mesh_path, cfg)
    mjcf = loader.get_mjcf_string()

    assert mjcf is not None


@pytest.mark.parametrize("robot_name", ["ergoCub", "ANYmal"])
def test_total_mass(robot_name):
    robot = ROBOTS[robot_name]
    controlled_joints = robot["controlled_joints"]
    urdf_path = robot["urdf_path"]
    mesh_path = robot["mesh_path"]

    control_modes = [ControlMode.TORQUE] * len(controlled_joints)
    stiffness = [0.0] * len(controlled_joints)
    damping = [0.0] * len(controlled_joints)
    cfg = URDFtoMuJoCoLoaderCfg(controlled_joints, control_modes, stiffness, damping)

    loader = URDFtoMuJoCoLoader.load_urdf(urdf_path, mesh_path, cfg)
    mjcf = loader.get_mjcf_string()

    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)

    total_mass_mj = sum(model.body_mass)

    # Test with iDynTree
    model_loader = idyntree.ModelLoader()
    model_loader.loadReducedModelFromFile(urdf_path, controlled_joints)
    idt_model = model_loader.model()
    total_mass_idt = idt_model.getTotalMass()

    assert total_mass_mj == pytest.approx(total_mass_idt, abs=1e-4)

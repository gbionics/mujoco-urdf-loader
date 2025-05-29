from mujoco_urdf_loader import URDFtoMuJoCoLoader, URDFtoMuJoCoLoaderCfg, ControlMode
import resolve_robotics_uri_py as rru

def test_load_urdf_into_mjcf():

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

    mjcf = loader.get_mjcf_string()

    assert mjcf is not None
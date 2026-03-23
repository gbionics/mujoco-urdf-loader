import xml.etree.ElementTree as ET

import pytest

from mujoco_urdf_loader.loader import (
    ControlMode,
    EqualityConstraintCfg,
    URDFtoMuJoCoLoader,
    URDFtoMuJoCoLoaderCfg,
)


def _make_joint_mjcf() -> ET.Element:
    return ET.fromstring(
        """
        <mujoco model="test_model">
            <worldbody>
                <body name="base">
                    <joint name="l_hip_pitch" type="hinge" range="-1 1"/>
                    <joint name="l_motor_1" type="hinge" range="-1 1"/>
                    <joint name="l_motor_rod_in_universal_0" type="hinge" range="-1 1"/>
                </body>
            </worldbody>
        </mujoco>
        """
    )


def _make_closed_chain_urdf() -> str:
    return """
    <robot name="closed_chain_test">
      <link name="world"/>
      <link name="base">
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="1.0"/>
          <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
        </inertial>
      </link>
      <joint name="world_to_base" type="fixed">
        <parent link="world"/>
        <child link="base"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
      </joint>

      <link name="left_ankle">
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="0.2"/>
          <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
        </inertial>
      </link>
      <joint name="l_hip_pitch" type="revolute">
        <parent link="base"/>
        <child link="left_ankle"/>
        <origin xyz="0 0.1 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.57" upper="1.57" effort="100" velocity="10"/>
      </joint>

      <link name="l_anklecr_in">
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="0.01"/>
          <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
        </inertial>
      </link>
      <joint name="l_anklecr_in_j" type="fixed">
        <parent link="left_ankle"/>
        <child link="l_anklecr_in"/>
        <origin xyz="0 0.02 0" rpy="0 0 0"/>
      </joint>
      <link name="l_anklecr_out">
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="0.01"/>
          <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
        </inertial>
      </link>
      <joint name="l_anklecr_out_j" type="fixed">
        <parent link="left_ankle"/>
        <child link="l_anklecr_out"/>
        <origin xyz="0 -0.02 0" rpy="0 0 0"/>
      </joint>
      <link name="l_rod_in_bottom">
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="0.01"/>
          <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
        </inertial>
      </link>
      <joint name="l_rod_in_bottom_j" type="fixed">
        <parent link="left_ankle"/>
        <child link="l_rod_in_bottom"/>
        <origin xyz="0 0.03 0" rpy="0 0 0"/>
      </joint>
      <link name="l_rod_out_bottom">
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="0.01"/>
          <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
        </inertial>
      </link>
      <joint name="l_rod_out_bottom_j" type="fixed">
        <parent link="left_ankle"/>
        <child link="l_rod_out_bottom"/>
        <origin xyz="0 -0.03 0" rpy="0 0 0"/>
      </joint>

      <link name="right_ankle">
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="0.2"/>
          <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
        </inertial>
      </link>
      <joint name="r_hip_pitch" type="revolute">
        <parent link="base"/>
        <child link="right_ankle"/>
        <origin xyz="0 -0.1 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.57" upper="1.57" effort="100" velocity="10"/>
      </joint>

      <link name="r_anklecr_in">
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="0.01"/>
          <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
        </inertial>
      </link>
      <joint name="r_anklecr_in_j" type="fixed">
        <parent link="right_ankle"/>
        <child link="r_anklecr_in"/>
        <origin xyz="0 0.02 0" rpy="0 0 0"/>
      </joint>
      <link name="r_anklecr_out">
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="0.01"/>
          <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
        </inertial>
      </link>
      <joint name="r_anklecr_out_j" type="fixed">
        <parent link="right_ankle"/>
        <child link="r_anklecr_out"/>
        <origin xyz="0 -0.02 0" rpy="0 0 0"/>
      </joint>
      <link name="r_rod_in_bottom">
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="0.01"/>
          <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
        </inertial>
      </link>
      <joint name="r_rod_in_bottom_j" type="fixed">
        <parent link="right_ankle"/>
        <child link="r_rod_in_bottom"/>
        <origin xyz="0 0.03 0" rpy="0 0 0"/>
      </joint>
      <link name="r_rod_out_bottom">
        <inertial>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <mass value="0.01"/>
          <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
        </inertial>
      </link>
      <joint name="r_rod_out_bottom_j" type="fixed">
        <parent link="right_ankle"/>
        <child link="r_rod_out_bottom"/>
        <origin xyz="0 -0.03 0" rpy="0 0 0"/>
      </joint>
    </robot>
    """


def _write_closed_chain_urdf(tmp_path) -> str:
    urdf_path = tmp_path / "closed_chain_test.urdf"
    urdf_path.write_text(_make_closed_chain_urdf(), encoding="utf-8")
    return str(urdf_path)


def test_observed_joints_must_be_provided():
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=None,
        actuated_joints=[],
    )

    with pytest.raises(ValueError, match="observed_joints must be provided"):
        URDFtoMuJoCoLoader(_make_joint_mjcf(), cfg)


def test_only_actuated_subset_gets_actuators():
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=["l_hip_pitch", "l_motor_1", "l_motor_rod_in_universal_0"],
        actuated_joints=["l_hip_pitch", "l_motor_1"],
        control_modes=[ControlMode.TORQUE, ControlMode.TORQUE],
    )
    loader = URDFtoMuJoCoLoader(_make_joint_mjcf(), cfg)

    actuator_joints = sorted(
        actuator.attrib["joint"] for actuator in loader.mjcf.findall(".//actuator/*")
    )
    assert actuator_joints == ["l_hip_pitch", "l_motor_1"]
    assert loader.mjcf.find(
        ".//actuator/*[@joint='l_motor_rod_in_universal_0']"
    ) is None


def test_observed_joint_without_actuator_remains_readable():
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=["l_hip_pitch", "l_motor_1", "l_motor_rod_in_universal_0"],
        actuated_joints=["l_hip_pitch", "l_motor_1"],
        control_modes=[ControlMode.TORQUE, ControlMode.TORQUE],
    )
    loader = URDFtoMuJoCoLoader(_make_joint_mjcf(), cfg)

    assert loader.mjcf.find(".//joint[@name='l_motor_rod_in_universal_0']") is not None


def test_actuated_joints_must_be_subset_of_observed_joints():
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=["l_hip_pitch"],
        actuated_joints=["not_observed"],
        control_modes=[ControlMode.TORQUE],
    )

    with pytest.raises(ValueError, match="subset of observed_joints"):
        URDFtoMuJoCoLoader(_make_joint_mjcf(), cfg)


def test_torque_mode_allows_missing_gains():
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=["l_hip_pitch"],
        actuated_joints=["l_hip_pitch"],
        control_modes=[ControlMode.TORQUE],
    )

    loader = URDFtoMuJoCoLoader(_make_joint_mjcf(), cfg)
    assert loader.mjcf.find(".//actuator/motor[@joint='l_hip_pitch']") is not None


def test_position_mode_requires_stiffness():
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=["l_hip_pitch"],
        actuated_joints=["l_hip_pitch"],
        control_modes=[ControlMode.POSITION],
        damping=[0.5],
    )

    with pytest.raises(ValueError, match="stiffness is required"):
        URDFtoMuJoCoLoader(_make_joint_mjcf(), cfg)


def test_position_mode_requires_damping():
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=["l_hip_pitch"],
        actuated_joints=["l_hip_pitch"],
        control_modes=[ControlMode.POSITION],
        stiffness=[20.0],
    )

    with pytest.raises(ValueError, match="damping is required"):
        URDFtoMuJoCoLoader(_make_joint_mjcf(), cfg)


def test_position_mode_gain_length_mismatch_raises():
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=["l_hip_pitch", "l_motor_1"],
        actuated_joints=["l_hip_pitch", "l_motor_1"],
        control_modes=[ControlMode.POSITION, ControlMode.POSITION],
        stiffness=[20.0],
        damping=[0.5, 0.6],
    )

    with pytest.raises(ValueError, match="Length of stiffness"):
        URDFtoMuJoCoLoader(_make_joint_mjcf(), cfg)


def test_position_mode_sets_kp_and_joint_damping():
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=["l_hip_pitch"],
        actuated_joints=["l_hip_pitch"],
        control_modes=[ControlMode.POSITION],
        stiffness=[42.0],
        damping=[0.7],
    )

    loader = URDFtoMuJoCoLoader(_make_joint_mjcf(), cfg)

    actuator = loader.mjcf.find(".//actuator/position[@joint='l_hip_pitch']")
    assert actuator is not None
    assert actuator.attrib["kp"] == "42.0"

    joint = loader.mjcf.find(".//joint[@name='l_hip_pitch']")
    assert joint is not None
    assert joint.attrib["damping"] == "0.7"


def test_joint_dynamics_are_set_from_observed_joint_lists():
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=["l_hip_pitch", "l_motor_1", "l_motor_rod_in_universal_0"],
        actuated_joints=["l_hip_pitch", "l_motor_1"],
        control_modes=[ControlMode.TORQUE, ControlMode.TORQUE],
        joint_damping=[0.1, 0.2, 0.3],
        joint_frictionloss=[1.0, 2.0, 3.0],
    )
    loader = URDFtoMuJoCoLoader(_make_joint_mjcf(), cfg)

    assert loader.mjcf.find(".//joint[@name='l_hip_pitch']").attrib["damping"] == "0.1"
    assert loader.mjcf.find(".//joint[@name='l_motor_1']").attrib["damping"] == "0.2"
    assert (
        loader.mjcf.find(".//joint[@name='l_motor_rod_in_universal_0']").attrib["damping"]
        == "0.3"
    )

    assert (
        loader.mjcf.find(".//joint[@name='l_hip_pitch']").attrib["frictionloss"] == "1.0"
    )
    assert loader.mjcf.find(".//joint[@name='l_motor_1']").attrib["frictionloss"] == "2.0"
    assert (
        loader.mjcf.find(".//joint[@name='l_motor_rod_in_universal_0']").attrib[
            "frictionloss"
        ]
        == "3.0"
    )


def test_joint_damping_length_must_match_observed_joints():
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=["l_hip_pitch", "l_motor_1"],
        actuated_joints=["l_hip_pitch"],
        control_modes=[ControlMode.TORQUE],
        joint_damping=[0.1],
    )

    with pytest.raises(ValueError, match="Length of joint_damping"):
        URDFtoMuJoCoLoader(_make_joint_mjcf(), cfg)


def test_joint_frictionloss_length_must_match_observed_joints():
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=["l_hip_pitch", "l_motor_1"],
        actuated_joints=["l_hip_pitch"],
        control_modes=[ControlMode.TORQUE],
        joint_frictionloss=[0.0],
    )

    with pytest.raises(ValueError, match="Length of joint_frictionloss"):
        URDFtoMuJoCoLoader(_make_joint_mjcf(), cfg)


def test_armature_defined_on_observed_joints_is_applied_only_to_actuated_joints():
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=["l_hip_pitch", "l_motor_rod_in_universal_0", "l_motor_1"],
        actuated_joints=["l_motor_1", "l_hip_pitch"],
        control_modes=[ControlMode.TORQUE, ControlMode.TORQUE],
        armature=[1.1, 2.2, 3.3],
    )
    normalized_cfg = URDFtoMuJoCoLoader._normalize_cfg(cfg)
    loader = URDFtoMuJoCoLoader(_make_joint_mjcf(), cfg)
    loader.set_armature(normalized_cfg.armature)

    assert loader.mjcf.find(".//joint[@name='l_motor_1']").attrib["armature"] == "3.3"
    assert loader.mjcf.find(".//joint[@name='l_hip_pitch']").attrib["armature"] == "1.1"
    assert (
        loader.mjcf.find(".//joint[@name='l_motor_rod_in_universal_0']").attrib.get(
            "armature"
        )
        is None
    )


def test_armature_can_be_specified_for_actuated_joints_only():
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=["l_hip_pitch", "l_motor_rod_in_universal_0", "l_motor_1"],
        actuated_joints=["l_motor_1", "l_hip_pitch"],
        control_modes=[ControlMode.TORQUE, ControlMode.TORQUE],
        armature=[9.0, 8.0],
    )
    normalized_cfg = URDFtoMuJoCoLoader._normalize_cfg(cfg)
    loader = URDFtoMuJoCoLoader(_make_joint_mjcf(), cfg)
    loader.set_armature(normalized_cfg.armature)

    assert loader.mjcf.find(".//joint[@name='l_motor_1']").attrib["armature"] == "9.0"
    assert loader.mjcf.find(".//joint[@name='l_hip_pitch']").attrib["armature"] == "8.0"


def test_armature_length_must_match_actuated_or_observed_joints():
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=["l_hip_pitch", "l_motor_rod_in_universal_0", "l_motor_1"],
        actuated_joints=["l_motor_1", "l_hip_pitch"],
        control_modes=[ControlMode.TORQUE, ControlMode.TORQUE],
        armature=[0.1],
    )

    with pytest.raises(ValueError, match="Length of armature"):
        URDFtoMuJoCoLoader(_make_joint_mjcf(), cfg)


def _closed_chain_cfg_with_constraints() -> URDFtoMuJoCoLoaderCfg:
    return URDFtoMuJoCoLoaderCfg(
        observed_joints=["l_hip_pitch", "r_hip_pitch"],
        actuated_joints=["l_hip_pitch", "r_hip_pitch"],
        control_modes=[ControlMode.TORQUE, ControlMode.TORQUE],
        all_missing_joints_as_sites=True,
        equality_constraints_cfg=[
            EqualityConstraintCfg(site1="l_anklecr_in", site2="l_rod_in_bottom"),
            EqualityConstraintCfg(site1="l_anklecr_out", site2="l_rod_out_bottom"),
            EqualityConstraintCfg(site1="r_anklecr_in", site2="r_rod_in_bottom"),
            EqualityConstraintCfg(site1="r_anklecr_out", site2="r_rod_out_bottom"),
        ],
    )


def test_closed_chain_connect_constraints_are_added_from_synthetic_urdf(tmp_path):
    urdf_path = _write_closed_chain_urdf(tmp_path)
    loader = URDFtoMuJoCoLoader.load_urdf(
        urdf_path=urdf_path,
        mesh_path=tmp_path,
        cfg=_closed_chain_cfg_with_constraints(),
    )

    # Missing fixed-link frames are recovered as sites and can be constrained.
    for site_name in (
        "l_anklecr_in",
        "l_anklecr_out",
        "l_rod_in_bottom",
        "l_rod_out_bottom",
        "r_anklecr_in",
        "r_anklecr_out",
        "r_rod_in_bottom",
        "r_rod_out_bottom",
    ):
        assert loader.mjcf.find(f".//site[@name='{site_name}']") is not None

    connects = loader.mjcf.findall(".//equality/connect")
    assert len(connects) == 4
    pairs = {(c.attrib["site1"], c.attrib["site2"]) for c in connects}
    assert pairs == {
        ("l_anklecr_in", "l_rod_in_bottom"),
        ("l_anklecr_out", "l_rod_out_bottom"),
        ("r_anklecr_in", "r_rod_in_bottom"),
        ("r_anklecr_out", "r_rod_out_bottom"),
    }


def test_closed_chain_missing_site_raises_from_synthetic_urdf(tmp_path):
    urdf_path = _write_closed_chain_urdf(tmp_path)
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=["l_hip_pitch", "r_hip_pitch"],
        actuated_joints=["l_hip_pitch", "r_hip_pitch"],
        control_modes=[ControlMode.TORQUE, ControlMode.TORQUE],
        all_missing_joints_as_sites=True,
    )
    loader = URDFtoMuJoCoLoader.load_urdf(
        urdf_path=urdf_path,
        mesh_path=tmp_path,
        cfg=cfg,
    )

    with pytest.raises(ValueError, match="not found"):
        loader.add_equality_constraints(
            [
                EqualityConstraintCfg(
                    site1="l_anklecr_in",
                    site2="missing_site",
                )
            ]
        )

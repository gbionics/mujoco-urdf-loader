"""Integration tests for spherical-to-ball-joint conversion pipeline.

Tests that the full collapse → MuJoCo load → ball joint conversion pipeline
works end-to-end, producing a valid MuJoCo model without zero-mass body errors.
"""

import xml.etree.ElementTree as ET

import mujoco
import pytest

from mujoco_urdf_loader.generator import load_urdf_into_mjcf
from mujoco_urdf_loader.mjcf_fcn import convert_hinge_to_ball_joints
from mujoco_urdf_loader.urdf_fcn import (
    add_mujoco_element,
    collapse_spherical_revolute_triplets,
    detect_spherical_joint_groups,
)


def _make_full_urdf():
    """Build a URDF with a floating root, a crank, and a spherical-joint rod.

    The spherical joint is represented as 3 revolute joints + 2 zero-mass
    dummy links, following the iDynTree convention.
    """
    urdf_str = """\
    <robot name="test_spherical">
      <link name="world"/>
      <link name="root_link">
        <inertial>
          <mass value="5.0"/>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
        </inertial>
        <visual>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <geometry><box size="0.1 0.1 0.1"/></geometry>
        </visual>
      </link>
      <joint name="floating_base" type="floating">
        <parent link="world"/>
        <child link="root_link"/>
      </joint>

      <link name="crank">
        <inertial>
          <mass value="0.3"/>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
        </inertial>
        <visual>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <geometry><cylinder radius="0.01" length="0.05"/></geometry>
        </visual>
      </link>
      <joint name="motor_joint" type="revolute">
        <origin xyz="0 0 -0.1" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <parent link="root_link"/>
        <child link="crank"/>
        <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
      </joint>

      <!-- Spherical joint triplet: crank -> dummy1 -> dummy2 -> rod -->
      <joint name="spherical_rev_my_rod_x" type="revolute">
        <origin xyz="0.05 0.02 0" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <parent link="crank"/>
        <child link="spherical_fake_my_rod_link1"/>
        <limit lower="-6.283" upper="6.283" effort="1000" velocity="1000"/>
        <dynamics damping="0" friction="0"/>
      </joint>
      <link name="spherical_fake_my_rod_link1">
        <inertial>
          <mass value="0"/>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
        </inertial>
      </link>
      <joint name="spherical_rev_my_rod_y" type="revolute">
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <parent link="spherical_fake_my_rod_link1"/>
        <child link="spherical_fake_my_rod_link2"/>
        <limit lower="-6.283" upper="6.283" effort="1000" velocity="1000"/>
        <dynamics damping="0" friction="0"/>
      </joint>
      <link name="spherical_fake_my_rod_link2">
        <inertial>
          <mass value="0"/>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
        </inertial>
      </link>
      <joint name="spherical_rev_my_rod_z" type="revolute">
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <parent link="spherical_fake_my_rod_link2"/>
        <child link="rod"/>
        <limit lower="-6.283" upper="6.283" effort="1000" velocity="1000"/>
        <dynamics damping="0" friction="0"/>
      </joint>
      <link name="rod">
        <inertial>
          <mass value="0.17"/>
          <origin xyz="0 0 -0.15" rpy="0 0 0"/>
          <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.0001"/>
        </inertial>
        <visual>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <geometry><cylinder radius="0.005" length="0.3"/></geometry>
        </visual>
      </link>
    </robot>
    """
    return ET.fromstring(urdf_str)


class TestSphericalBallConversionPipeline:
    """Test the full pipeline: detect → collapse → load → convert."""

    def test_uncollapsed_urdf_fails_in_mujoco(self):
        """The original URDF with zero-mass dummy links should fail in MuJoCo."""
        urdf = _make_full_urdf()
        urdf_str = ET.tostring(urdf, encoding="unicode")
        with pytest.raises(Exception):
            mujoco.MjModel.from_xml_string(urdf_str)

    def test_collapsed_urdf_loads_in_mujoco(self):
        """After collapsing, the URDF should load in MuJoCo without errors."""
        urdf = _make_full_urdf()
        controlled = [
            "motor_joint",
            "spherical_rev_my_rod_x",
            "spherical_rev_my_rod_y",
            "spherical_rev_my_rod_z",
        ]
        groups = detect_spherical_joint_groups(controlled)
        collapsed_urdf, ball_map = collapse_spherical_revolute_triplets(urdf, groups)

        # Should be loadable by MuJoCo (no zero-mass error)
        urdf_str = ET.tostring(collapsed_urdf, encoding="unicode")
        model = mujoco.MjModel.from_xml_string(urdf_str)
        assert model is not None

    def test_full_pipeline_produces_ball_joint(self):
        """Full pipeline: collapse → load_urdf_into_mjcf → convert to ball."""
        urdf = _make_full_urdf()
        controlled = [
            "motor_joint",
            "spherical_rev_my_rod_x",
            "spherical_rev_my_rod_y",
            "spherical_rev_my_rod_z",
        ]
        groups = detect_spherical_joint_groups(controlled)
        collapsed_urdf, ball_map = collapse_spherical_revolute_triplets(urdf, groups)

        # Add mujoco compiler element (no mesh dir needed for this URDF)
        mj_elem = ET.SubElement(collapsed_urdf, "mujoco")
        compiler = ET.SubElement(mj_elem, "compiler")
        compiler.set("angle", "radian")

        mjcf = load_urdf_into_mjcf(collapsed_urdf)

        # Convert placeholder to ball
        convert_hinge_to_ball_joints(mjcf, ball_map)

        # Verify the ball joint exists
        ball_joint = mjcf.find(".//joint[@name='my_rod_ball']")
        assert ball_joint is not None
        assert ball_joint.attrib["type"] == "ball"

        # Verify damping and armature are set for stability
        assert float(ball_joint.attrib["damping"]) > 0
        assert float(ball_joint.attrib["armature"]) > 0

        # Verify no dummy bodies remain
        body_names = {b.attrib["name"] for b in mjcf.findall(".//body")}
        assert "spherical_fake_my_rod_link1" not in body_names
        assert "spherical_fake_my_rod_link2" not in body_names

        # Rod body should exist
        assert "rod" in body_names

    def test_full_pipeline_model_simulates(self):
        """The final model should be simulatable."""
        urdf = _make_full_urdf()
        controlled = [
            "motor_joint",
            "spherical_rev_my_rod_x",
            "spherical_rev_my_rod_y",
            "spherical_rev_my_rod_z",
        ]
        groups = detect_spherical_joint_groups(controlled)
        collapsed_urdf, ball_map = collapse_spherical_revolute_triplets(urdf, groups)

        mj_elem = ET.SubElement(collapsed_urdf, "mujoco")
        compiler = ET.SubElement(mj_elem, "compiler")
        compiler.set("angle", "radian")

        mjcf = load_urdf_into_mjcf(collapsed_urdf)
        convert_hinge_to_ball_joints(mjcf, ball_map, damping=0.01, armature=0.001)

        mjcf_str = ET.tostring(mjcf, encoding="unicode")
        model = mujoco.MjModel.from_xml_string(mjcf_str)
        data = mujoco.MjData(model)

        # Run simulation long enough to catch instability
        for _ in range(1000):
            mujoco.mj_step(model, data)

        # Should not crash or produce NaN / Inf
        import numpy as np
        assert np.all(np.isfinite(data.qpos)), "qpos contains NaN/Inf after simulation"

    def test_no_spherical_groups_is_noop(self):
        """When there are no spherical groups, the pipeline is a no-op."""
        urdf = _make_full_urdf()
        controlled = ["motor_joint"]
        groups = detect_spherical_joint_groups(controlled)
        assert len(groups) == 0

        # collapse with empty groups should not modify the URDF
        collapsed_urdf, ball_map = collapse_spherical_revolute_triplets(urdf, groups)
        assert ball_map == {}
        # All links should still be present
        link_names = {l.attrib["name"] for l in collapsed_urdf.findall(".//link")}
        assert "spherical_fake_my_rod_link1" in link_names

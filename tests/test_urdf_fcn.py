import xml.etree.ElementTree as ET

import pytest

from mujoco_urdf_loader.urdf_fcn import (
    add_mujoco_element,
    get_joint_limits,
    get_mesh_path,
    get_robot_urdf,
    resolve_mesh_filenames,
    remove_gazebo_elements,
    remove_links_and_joints_by_keep_list,
    remove_links_and_joints_by_remove_list,
    detect_spherical_joint_groups,
    collapse_spherical_revolute_triplets,
)


def test_get_robot_urdf():
    robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")
    assert robot_urdf is not None
    assert robot_urdf.tag == "robot"
    assert robot_urdf.attrib["name"] == "ergoCub"
    return


def test_get_mesh_path():
    robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")
    mesh_path = get_mesh_path(robot_urdf)
    assert mesh_path is not None
    assert mesh_path.is_dir()
    return


def test_remove_gazebo_elements():
    robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")
    robot_urdf = remove_gazebo_elements(robot_urdf)
    assert robot_urdf is not None
    assert len(robot_urdf.findall(".//gazebo")) == 0
    return


def test_remove_links_and_joints():
    robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")
    to_remove = [
        "leg",
        "foot",
        "ankle",
        "hip",
        "knee",
        "sole",
    ]
    robot_urdf = remove_links_and_joints_by_remove_list(robot_urdf, to_remove)
    assert robot_urdf is not None

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
    robot_urdf = remove_links_and_joints_by_keep_list(robot_urdf, to_keep)
    assert robot_urdf is not None
    return


def test_add_mujoco_element():
    robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")
    mesh_path = get_mesh_path(robot_urdf)
    robot_urdf = add_mujoco_element(robot_urdf, mesh_path)
    assert robot_urdf is not None
    assert len(robot_urdf.findall(".//mujoco")) == 1
    return


def test_resolve_mesh_filenames():
    robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")
    robot_urdf = resolve_mesh_filenames(robot_urdf)
    mesh = robot_urdf.find(".//mesh")
    assert mesh is not None
    assert not mesh.attrib["filename"].startswith("package://")
    assert mesh.attrib["filename"].endswith(".stl")
    return


def test_get_joint_limits():
    robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")
    joint_limits = get_joint_limits(robot_urdf)
    assert joint_limits is not None
    assert len(joint_limits) == 57
    return


# ---------------------------------------------------------------------------
# Tests for spherical joint detection and collapse
# ---------------------------------------------------------------------------

def _make_spherical_urdf():
    """Build a minimal URDF with one spherical-joint triplet (3 revolute + 2 dummy links)."""
    urdf_str = """
    <robot name="test_robot">
      <link name="root_link">
        <inertial>
          <mass value="1.0"/>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
        </inertial>
      </link>
      <link name="crank">
        <inertial>
          <mass value="0.5"/>
          <origin xyz="0 0 0" rpy="0 0 0"/>
          <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
        </inertial>
      </link>
      <joint name="crank_joint" type="revolute">
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <parent link="root_link"/>
        <child link="crank"/>
        <limit lower="-1" upper="1" effort="100" velocity="10"/>
      </joint>
      <!-- Spherical joint triplet -->
      <joint name="spherical_rev_my_rod_x" type="revolute">
        <origin xyz="0.05 0.02 0" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <parent link="crank"/>
        <child link="spherical_fake_my_rod_link1"/>
        <limit lower="-6.28" upper="6.28" effort="1000" velocity="1000"/>
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
        <limit lower="-6.28" upper="6.28" effort="1000" velocity="1000"/>
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
        <limit lower="-6.28" upper="6.28" effort="1000" velocity="1000"/>
        <dynamics damping="0" friction="0"/>
      </joint>
      <link name="rod">
        <inertial>
          <mass value="0.2"/>
          <origin xyz="0 0 -0.15" rpy="0 0 0"/>
          <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.0001"/>
        </inertial>
      </link>
    </robot>
    """
    return ET.fromstring(urdf_str)


def test_detect_spherical_joint_groups():
    joints = [
        "crank_joint",
        "spherical_rev_my_rod_x",
        "spherical_rev_my_rod_y",
        "spherical_rev_my_rod_z",
    ]
    groups = detect_spherical_joint_groups(joints)
    assert len(groups) == 1
    assert groups[0]["base_name"] == "my_rod"
    assert groups[0]["joint_x"] == "spherical_rev_my_rod_x"
    assert groups[0]["joint_y"] == "spherical_rev_my_rod_y"
    assert groups[0]["joint_z"] == "spherical_rev_my_rod_z"


def test_detect_spherical_joint_groups_multiple():
    joints = [
        "spherical_rev_r_motor_rod_in_x",
        "spherical_rev_r_motor_rod_in_y",
        "spherical_rev_r_motor_rod_in_z",
        "spherical_rev_l_motor_rod_in_x",
        "spherical_rev_l_motor_rod_in_y",
        "spherical_rev_l_motor_rod_in_z",
        "l_hip_pitch",
    ]
    groups = detect_spherical_joint_groups(joints)
    assert len(groups) == 2
    base_names = {g["base_name"] for g in groups}
    assert base_names == {"r_motor_rod_in", "l_motor_rod_in"}


def test_detect_no_spherical_groups():
    joints = ["l_hip_pitch", "r_hip_pitch", "l_knee", "r_knee"]
    groups = detect_spherical_joint_groups(joints)
    assert len(groups) == 0


def test_detect_incomplete_triplet():
    """Incomplete triplet (missing z) should not be detected."""
    joints = ["spherical_rev_my_rod_x", "spherical_rev_my_rod_y"]
    groups = detect_spherical_joint_groups(joints)
    assert len(groups) == 0


def test_collapse_spherical_revolute_triplets():
    urdf = _make_spherical_urdf()
    groups = detect_spherical_joint_groups([
        "crank_joint",
        "spherical_rev_my_rod_x",
        "spherical_rev_my_rod_y",
        "spherical_rev_my_rod_z",
    ])

    collapsed_urdf, ball_map = collapse_spherical_revolute_triplets(urdf, groups)

    # Dummy links should be removed
    link_names = {l.attrib["name"] for l in collapsed_urdf.findall(".//link")}
    assert "spherical_fake_my_rod_link1" not in link_names
    assert "spherical_fake_my_rod_link2" not in link_names

    # _y and _z joints should be removed
    joint_names = {j.attrib["name"] for j in collapsed_urdf.findall(".//joint")}
    assert "spherical_rev_my_rod_y" not in joint_names
    assert "spherical_rev_my_rod_z" not in joint_names

    # _x joint should still exist as a continuous joint
    assert "spherical_rev_my_rod_x" in joint_names
    jx = collapsed_urdf.find(".//joint[@name='spherical_rev_my_rod_x']")
    assert jx.attrib["type"] == "continuous"
    # Child should be the final child (rod), not a dummy link
    assert jx.find("child").attrib["link"] == "rod"
    # Parent should still be crank
    assert jx.find("parent").attrib["link"] == "crank"

    # The ball_map should map placeholder -> base_name
    assert ball_map == {"spherical_rev_my_rod_x": "my_rod"}


def test_collapse_preserves_rod_link():
    urdf = _make_spherical_urdf()
    groups = detect_spherical_joint_groups([
        "crank_joint",
        "spherical_rev_my_rod_x",
        "spherical_rev_my_rod_y",
        "spherical_rev_my_rod_z",
    ])

    collapsed_urdf, _ = collapse_spherical_revolute_triplets(urdf, groups)

    # Rod link should still exist with its original inertia
    rod = collapsed_urdf.find(".//link[@name='rod']")
    assert rod is not None
    mass = rod.find("inertial/mass")
    assert float(mass.attrib["value"]) == pytest.approx(0.2)


def test_collapse_preserves_non_spherical_joints():
    urdf = _make_spherical_urdf()
    groups = detect_spherical_joint_groups([
        "crank_joint",
        "spherical_rev_my_rod_x",
        "spherical_rev_my_rod_y",
        "spherical_rev_my_rod_z",
    ])

    collapsed_urdf, _ = collapse_spherical_revolute_triplets(urdf, groups)

    # crank_joint should be untouched
    crank_joint = collapsed_urdf.find(".//joint[@name='crank_joint']")
    assert crank_joint is not None
    assert crank_joint.attrib["type"] == "revolute"


def test_collapse_preserves_origin():
    urdf = _make_spherical_urdf()
    groups = detect_spherical_joint_groups([
        "crank_joint",
        "spherical_rev_my_rod_x",
        "spherical_rev_my_rod_y",
        "spherical_rev_my_rod_z",
    ])

    collapsed_urdf, _ = collapse_spherical_revolute_triplets(urdf, groups)

    jx = collapsed_urdf.find(".//joint[@name='spherical_rev_my_rod_x']")
    origin = jx.find("origin")
    assert origin is not None
    assert origin.attrib["xyz"] == "0.05 0.02 0"

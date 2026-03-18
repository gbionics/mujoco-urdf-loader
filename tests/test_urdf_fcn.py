import xml.etree.ElementTree as ET

import pytest

from mujoco_urdf_loader.urdf_fcn import (
    add_mujoco_element,
    get_joint_limits,
    get_mesh_path,
    get_robot_urdf,
    remove_gazebo_elements,
    remove_links_and_joints_by_keep_list,
    remove_links_and_joints_by_remove_list,
    find_spherical_revolute_joints_in_urdf,
    set_min_mass_for_zero_mass_links,
    _is_zero_mass_link,
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


def test_get_joint_limits():
    robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")
    joint_limits = get_joint_limits(robot_urdf)
    assert joint_limits is not None
    assert len(joint_limits) == 57
    return


# ---------------------------------------------------------------------------
# Spherical revolute joint detection in URDF
# ---------------------------------------------------------------------------


def _build_urdf_with_spherical_chain():
    """Build a minimal URDF with one 3-revolute spherical chain."""
    return ET.fromstring(
        """<robot name="test">
  <link name="parent">
    <inertial><mass value="1.0"/></inertial>
  </link>
  <joint name="spherical_rev_foo_x" type="revolute">
    <parent link="parent"/><child link="spherical_fake_foo_link1"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/><axis xyz="1 0 0"/>
  </joint>
  <link name="spherical_fake_foo_link1">
    <inertial><mass value="0"/></inertial>
  </link>
  <joint name="spherical_rev_foo_y" type="revolute">
    <parent link="spherical_fake_foo_link1"/><child link="spherical_fake_foo_link2"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
  </joint>
  <link name="spherical_fake_foo_link2">
    <inertial><mass value="0"/></inertial>
  </link>
  <joint name="spherical_rev_foo_z" type="revolute">
    <parent link="spherical_fake_foo_link2"/><child link="child"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
  </joint>
  <link name="child">
    <inertial><mass value="0.5"/></inertial>
  </link>
</robot>"""
    )


def test_find_spherical_joints_basic():
    urdf = _build_urdf_with_spherical_chain()
    names = find_spherical_revolute_joints_in_urdf(urdf)
    assert names == [
        "spherical_rev_foo_x",
        "spherical_rev_foo_y",
        "spherical_rev_foo_z",
    ]


def test_find_spherical_joints_no_match():
    urdf = ET.fromstring(
        """<robot name="test">
  <link name="a"><inertial><mass value="1.0"/></inertial></link>
  <joint name="j" type="revolute">
    <parent link="a"/><child link="b"/><axis xyz="1 0 0"/>
  </joint>
  <link name="b"><inertial><mass value="2.0"/></inertial></link>
</robot>"""
    )
    assert find_spherical_revolute_joints_in_urdf(urdf) == []


def test_find_spherical_joints_intermediate_has_mass():
    """Intermediate link with non-zero mass should not match."""
    urdf = _build_urdf_with_spherical_chain()
    # Give fake_link1 real mass
    link1 = urdf.find(".//link[@name='spherical_fake_foo_link1']")
    link1.find("inertial/mass").set("value", "5.0")
    assert find_spherical_revolute_joints_in_urdf(urdf) == []


def test_find_spherical_joints_intermediate_has_visual():
    """Intermediate link with visual should not match."""
    urdf = _build_urdf_with_spherical_chain()
    link1 = urdf.find(".//link[@name='spherical_fake_foo_link1']")
    ET.SubElement(link1, "visual")
    assert find_spherical_revolute_joints_in_urdf(urdf) == []


def test_find_spherical_joints_multiple_chains():
    """Two independent spherical chains."""
    urdf = ET.fromstring(
        """<robot name="test">
  <link name="root"><inertial><mass value="5"/></inertial></link>
  <joint name="a_x" type="revolute"><parent link="root"/><child link="fa1"/><axis xyz="1 0 0"/></joint>
  <link name="fa1"><inertial><mass value="0"/></inertial></link>
  <joint name="a_y" type="revolute"><parent link="fa1"/><child link="fa2"/><axis xyz="0 1 0"/></joint>
  <link name="fa2"><inertial><mass value="0"/></inertial></link>
  <joint name="a_z" type="revolute"><parent link="fa2"/><child link="child_a"/><axis xyz="0 0 1"/></joint>
  <link name="child_a"><inertial><mass value="1"/></inertial></link>
  <joint name="b_x" type="revolute"><parent link="root"/><child link="fb1"/><axis xyz="1 0 0"/></joint>
  <link name="fb1"><inertial><mass value="0"/></inertial></link>
  <joint name="b_y" type="revolute"><parent link="fb1"/><child link="fb2"/><axis xyz="0 1 0"/></joint>
  <link name="fb2"><inertial><mass value="0"/></inertial></link>
  <joint name="b_z" type="revolute"><parent link="fb2"/><child link="child_b"/><axis xyz="0 0 1"/></joint>
  <link name="child_b"><inertial><mass value="2"/></inertial></link>
</robot>"""
    )
    names = find_spherical_revolute_joints_in_urdf(urdf)
    assert len(names) == 6
    assert set(names) == {"a_x", "a_y", "a_z", "b_x", "b_y", "b_z"}


# ---------------------------------------------------------------------------
# Zero-mass link utilities
# ---------------------------------------------------------------------------


def test_is_zero_mass_link_no_inertial():
    link = ET.fromstring('<link name="a"/>')
    assert _is_zero_mass_link(link) is True


def test_is_zero_mass_link_zero():
    link = ET.fromstring('<link name="a"><inertial><mass value="0"/></inertial></link>')
    assert _is_zero_mass_link(link) is True


def test_is_zero_mass_link_nonzero():
    link = ET.fromstring('<link name="a"><inertial><mass value="1.5"/></inertial></link>')
    assert _is_zero_mass_link(link) is False


def test_set_min_mass():
    urdf = ET.fromstring(
        """<robot name="test">
  <link name="a"><inertial><mass value="0"/><inertia ixx="0" iyy="0" izz="0" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="b"><inertial><mass value="5.0"/></inertial></link>
  <link name="c"/>
</robot>"""
    )
    set_min_mass_for_zero_mass_links(urdf)
    a = urdf.find(".//link[@name='a']")
    assert float(a.find("inertial/mass").get("value")) == 1e-8
    b = urdf.find(".//link[@name='b']")
    assert float(b.find("inertial/mass").get("value")) == 5.0
    c = urdf.find(".//link[@name='c']")
    assert float(c.find("inertial/mass").get("value")) == 1e-8

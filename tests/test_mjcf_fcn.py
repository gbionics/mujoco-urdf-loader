import xml.etree.ElementTree as ET

import pytest

from mujoco_urdf_loader.mjcf_fcn import (
    add_camera,
    add_joint_eq,
    add_joint_pos_sensor,
    add_joint_vel_sensor,
    add_new_worldbody,
    add_position_actuator,
    convert_spherical_joints_to_ball,
    find_spherical_joint_patterns,
    separate_left_right_collision_groups,
    set_collision_groups,
    set_joint_damping,
    _derive_ball_joint_name,
    _is_zero_mass_body,
)


def test_add_new_worldbody():
    mjcf = ET.Element("mujoco")
    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "test")
    mjcf = add_new_worldbody(mjcf)

    assert mjcf is not None
    # check that there is still only one worldbody element
    assert len(mjcf.findall(".//worldbody")) == 1
    # check that there is now a new body element in the worldbody
    assert len(mjcf.findall(".//worldbody/body")) == 1


def test_add_position_actuator():
    mjcf = ET.Element("mujoco")

    joint = ET.SubElement(mjcf, "joint")
    joint.set("name", "joint1")

    mjcf = add_position_actuator(mjcf, joint.attrib["name"])

    assert mjcf is not None
    # check that there is now an actuator element in the mjcf
    assert len(mjcf.findall(".//actuator")) == 1
    # check that there is a position actuator element in the actuator
    assert len(mjcf.findall(".//actuator/position")) == 1
    # check that the actuator is connected to the joint
    assert mjcf.find(".//actuator/position").attrib["joint"] == "joint1"

    # test that adding another actuator does not add another actuator element\
    joint = ET.SubElement(mjcf, "joint")
    joint.set("name", "joint2")

    mjcf = add_position_actuator(mjcf, joint.attrib["name"])

    assert len(mjcf.findall(".//actuator")) == 1
    assert len(mjcf.findall(".//actuator/position")) == 2


def test_add_joint_pos_sensor():
    mjcf = ET.Element("mujoco")

    joint = ET.SubElement(mjcf, "joint")
    joint.set("name", "joint1")

    mjcf = add_joint_pos_sensor(mjcf, joint.attrib["name"])

    assert mjcf is not None
    # check that there is now a sensor element in the mjcf
    assert len(mjcf.findall(".//sensor")) == 1
    # check that there is a joint pos sensor element in the sensor
    assert len(mjcf.findall(".//sensor/jointpos")) == 1
    # check that the sensor is connected to the joint
    assert mjcf.find(".//sensor/jointpos").attrib["joint"] == "joint1"

    # test that adding another sensor does not add another sensor element
    joint = ET.SubElement(mjcf, "joint")
    joint.set("name", "joint2")

    mjcf = add_joint_pos_sensor(mjcf, joint.attrib["name"])

    assert len(mjcf.findall(".//sensor")) == 1
    assert len(mjcf.findall(".//sensor/jointpos")) == 2


def test_add_joint_vel_sensor():
    mjcf = ET.Element("mujoco")

    joint = ET.SubElement(mjcf, "joint")
    joint.set("name", "joint1")

    mjcf = add_joint_vel_sensor(mjcf, joint.attrib["name"])

    assert mjcf is not None
    # check that there is now a sensor element in the mjcf
    assert len(mjcf.findall(".//sensor")) == 1
    # check that there is a joint vel sensor element in the sensor
    assert len(mjcf.findall(".//sensor/jointvel")) == 1
    # check that the sensor is connected to the joint
    assert mjcf.find(".//sensor/jointvel").attrib["joint"] == "joint1"

    # test that adding another sensor does not add another sensor element
    joint = ET.SubElement(mjcf, "joint")
    joint.set("name", "joint2")

    mjcf = add_joint_vel_sensor(mjcf, joint.attrib["name"])

    assert len(mjcf.findall(".//sensor")) == 1
    assert len(mjcf.findall(".//sensor/jointvel")) == 2


def test_add_joint_eq():
    mjcf = ET.Element("mujoco")

    joint1 = ET.SubElement(mjcf, "joint")
    joint1.set("name", "joint1")
    joint2 = ET.SubElement(mjcf, "joint")
    joint2.set("name", "joint2")

    mjcf = add_joint_eq(mjcf, joint1.attrib["name"], joint2.attrib["name"])

    assert mjcf is not None
    # check that there is now an equality element in the mjcf
    assert len(mjcf.findall(".//equality")) == 1
    assert len(mjcf.findall(".//equality/joint")) == 1
    # check that the equality element is connected to the joints
    assert mjcf.find(".//equality/joint").attrib["joint1"] == "joint1"
    assert mjcf.find(".//equality/joint").attrib["joint2"] == "joint2"

    # test that adding another equality does not add another equality element
    joint1 = ET.SubElement(mjcf, "joint")
    joint1.set("name", "joint3")
    joint2 = ET.SubElement(mjcf, "joint")
    joint2.set("name", "joint4")

    mjcf = add_joint_eq(mjcf, joint1.attrib["name"], joint2.attrib["name"])

    assert len(mjcf.findall(".//equality")) == 1
    assert len(mjcf.findall(".//equality/joint")) == 2


def test_add_camera():
    mjcf = ET.Element("mujoco")
    body = ET.SubElement(mjcf, "body")

    add_camera(body)

    # check that there is now a camera element in the mjcf under the body
    assert len(mjcf.findall(".//body/camera")) == 1


def test_set_collision_group():
    mjcf = ET.Element("mujoco")
    geom = ET.SubElement(mjcf, "geom")
    geom.set("mesh", "test")

    mjcf = set_collision_groups(mjcf, idx="test", group=1, affinity=1)

    # check the contype and conaffinity of the geom
    assert mjcf.find(".//geom").attrib["contype"] == "1"
    assert mjcf.find(".//geom").attrib["conaffinity"] == "1"


def test_separate_left_right_collision_groups():
    mjcf = ET.Element("mujoco")
    body = ET.SubElement(mjcf, "body")
    geom = ET.SubElement(body, "geom")
    geom.set("mesh", "_r_test")
    geom = ET.SubElement(body, "geom")
    geom.set("mesh", "_l_test")

    separate_left_right_collision_groups(mjcf, r_group=1, l_group=2, r_aff=1, l_aff=2)

    # check the contype and conaffinity of the geom
    assert mjcf.find(".//geom[@mesh='_r_test']").attrib["contype"] == "1"
    assert mjcf.find(".//geom[@mesh='_r_test']").attrib["conaffinity"] == "1"
    assert mjcf.find(".//geom[@mesh='_l_test']").attrib["contype"] == "2"
    assert mjcf.find(".//geom[@mesh='_l_test']").attrib["conaffinity"] == "2"


def test_set_joint_damping():
    mjcf = ET.Element("mujoco")
    body = ET.SubElement(mjcf, "body")
    joint = ET.SubElement(body, "joint")
    joint.set("name", "test")

    set_joint_damping(mjcf, damping=1.0)

    print(joint.attrib)

    assert joint.attrib["damping"] == "1.0"
    return


# ---------------------------------------------------------------------------
# Helpers for spherical joint tests
# ---------------------------------------------------------------------------


def _build_spherical_chain_mjcf():
    """Build a mock MJCF with a single 3-hinge spherical joint pattern."""
    return ET.fromstring(
        """<mujoco>
  <worldbody>
    <body name="parent_link" pos="0 0 0">
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="spherical_fake_test_joint_link1" pos="0.1 0.2 0.3">
        <joint name="spherical_rev_test_joint_x" type="hinge" axis="1 0 0" pos="0 0 0" damping="0"/>
        <body name="spherical_fake_test_joint_link2">
          <joint name="spherical_rev_test_joint_y" type="hinge" axis="0 1 0" pos="0 0 0" damping="0"/>
          <body name="child_link" pos="0 0 0">
            <inertial mass="0.5" pos="0.01 0.02 0.03" diaginertia="0.01 0.01 0.01"/>
            <joint name="spherical_rev_test_joint_z" type="hinge" axis="0 0 1" pos="0 0 0" damping="0"/>
            <geom type="mesh" mesh="child_mesh"/>
          </body>
        </body>
      </body>
      <body name="normal_child" pos="0.5 0 0">
        <inertial mass="2.0" pos="0 0 0" diaginertia="0.2 0.2 0.2"/>
        <joint name="normal_joint" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.2 0.2 0.2"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""
    )


def _compute_total_mass(mjcf):
    """Sum the mass of all bodies in the MJCF."""
    total = 0.0
    for body in mjcf.iter("body"):
        inertial = body.find("inertial")
        if inertial is not None:
            total += float(inertial.get("mass", "0"))
    return total


def _get_all_body_names(mjcf):
    """Get all body names in the MJCF."""
    return {body.get("name") for body in mjcf.iter("body")}


def _get_all_joint_names_and_types(mjcf):
    """Get dict of joint_name -> type."""
    return {j.get("name"): j.get("type") for j in mjcf.iter("joint")}


# ---------------------------------------------------------------------------
# Tests: _is_zero_mass_body
# ---------------------------------------------------------------------------


def test_is_zero_mass_body_no_inertial():
    body = ET.Element("body")
    assert _is_zero_mass_body(body) is True


def test_is_zero_mass_body_zero_mass():
    body = ET.Element("body")
    inertial = ET.SubElement(body, "inertial")
    inertial.set("mass", "0")
    assert _is_zero_mass_body(body) is True


def test_is_zero_mass_body_small_mass():
    body = ET.Element("body")
    inertial = ET.SubElement(body, "inertial")
    inertial.set("mass", "1e-10")
    assert _is_zero_mass_body(body) is True


def test_is_zero_mass_body_normal_mass():
    body = ET.Element("body")
    inertial = ET.SubElement(body, "inertial")
    inertial.set("mass", "1.5")
    assert _is_zero_mass_body(body) is False


# ---------------------------------------------------------------------------
# Tests: _derive_ball_joint_name
# ---------------------------------------------------------------------------


def test_derive_ball_name_strip_x():
    assert _derive_ball_joint_name("spherical_rev_test_x") == "spherical_rev_test"


def test_derive_ball_name_strip_x_real():
    assert (
        _derive_ball_joint_name("spherical_rev_r_motor_rod_in_x")
        == "spherical_rev_r_motor_rod_in"
    )


def test_derive_ball_name_fallback_rev():
    assert _derive_ball_joint_name("something_rev_test") == "something"


def test_derive_ball_name_fallback_append():
    assert _derive_ball_joint_name("my_joint") == "my_joint_ball"


# ---------------------------------------------------------------------------
# Tests: find_spherical_joint_patterns
# ---------------------------------------------------------------------------


def test_find_patterns_single():
    mjcf = _build_spherical_chain_mjcf()
    patterns = find_spherical_joint_patterns(mjcf)
    assert len(patterns) == 1
    p = patterns[0]
    assert p["first_fake"].get("name") == "spherical_fake_test_joint_link1"
    assert p["second_fake"].get("name") == "spherical_fake_test_joint_link2"
    assert p["real_body"].get("name") == "child_link"
    assert p["joint1"].get("name") == "spherical_rev_test_joint_x"
    assert p["joint2"].get("name") == "spherical_rev_test_joint_y"
    assert p["joint3"].get("name") == "spherical_rev_test_joint_z"


def test_find_patterns_no_false_positive():
    mjcf = ET.fromstring(
        """<mujoco>
  <worldbody>
    <body name="a" pos="0 0 0">
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="b" pos="0.1 0 0">
        <inertial mass="2.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
        <joint name="j1" type="hinge" axis="1 0 0"/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""
    )
    assert len(find_spherical_joint_patterns(mjcf)) == 0


def test_find_patterns_no_match_intermediate_has_mass():
    mjcf = ET.fromstring(
        """<mujoco>
  <worldbody>
    <body name="parent" pos="0 0 0">
      <body name="fake1" pos="0.1 0 0">
        <inertial mass="1.0" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
        <joint name="j1" type="hinge" axis="1 0 0"/>
        <body name="fake2">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <body name="child">
            <inertial mass="0.5" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
            <joint name="j3" type="hinge" axis="0 0 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""
    )
    assert len(find_spherical_joint_patterns(mjcf)) == 0


def test_find_patterns_no_match_intermediate_has_geom():
    mjcf = ET.fromstring(
        """<mujoco>
  <worldbody>
    <body name="parent" pos="0 0 0">
      <body name="fake1" pos="0.1 0 0">
        <joint name="j1" type="hinge" axis="1 0 0"/>
        <geom type="box" size="0.1 0.1 0.1"/>
        <body name="fake2">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <body name="child">
            <inertial mass="0.5" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
            <joint name="j3" type="hinge" axis="0 0 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""
    )
    assert len(find_spherical_joint_patterns(mjcf)) == 0


# ---------------------------------------------------------------------------
# Tests: convert_spherical_joints_to_ball
# ---------------------------------------------------------------------------


def test_convert_basic():
    mjcf = _build_spherical_chain_mjcf()
    convert_spherical_joints_to_ball(mjcf)

    # Ball joint exists
    ball_joints = [j for j in mjcf.iter("joint") if j.get("type") == "ball"]
    assert len(ball_joints) == 1
    assert ball_joints[0].get("name") == "spherical_rev_test_joint"

    # Fake bodies removed
    body_names = _get_all_body_names(mjcf)
    assert "spherical_fake_test_joint_link1" not in body_names
    assert "spherical_fake_test_joint_link2" not in body_names

    # child_link is now a direct child of parent_link
    parent = mjcf.find(".//body[@name='parent_link']")
    child_names = [b.get("name") for b in parent.findall("body")]
    assert "child_link" in child_names

    # Position transferred
    child = mjcf.find(".//body[@name='child_link']")
    assert child.get("pos") == "0.1 0.2 0.3"


def test_convert_preserves_mass():
    mjcf = _build_spherical_chain_mjcf()
    mass_before = _compute_total_mass(mjcf)
    convert_spherical_joints_to_ball(mjcf)
    mass_after = _compute_total_mass(mjcf)
    assert abs(mass_before - mass_after) < 1e-9


def test_convert_preserves_normal_bodies():
    mjcf = _build_spherical_chain_mjcf()
    convert_spherical_joints_to_ball(mjcf)

    normal = mjcf.find(".//body[@name='normal_child']")
    assert normal is not None
    assert normal.find("joint").get("type") == "hinge"
    assert normal.find("joint").get("name") == "normal_joint"


def test_convert_preserves_child_body_content():
    mjcf = _build_spherical_chain_mjcf()
    convert_spherical_joints_to_ball(mjcf)

    child = mjcf.find(".//body[@name='child_link']")
    inertial = child.find("inertial")
    assert inertial is not None
    assert inertial.get("mass") == "0.5"
    geom = child.find("geom")
    assert geom is not None
    assert geom.get("mesh") == "child_mesh"


def test_convert_removes_hinge_joints():
    mjcf = _build_spherical_chain_mjcf()
    convert_spherical_joints_to_ball(mjcf)

    joints = _get_all_joint_names_and_types(mjcf)
    assert "spherical_rev_test_joint_x" not in joints
    assert "spherical_rev_test_joint_y" not in joints
    assert "spherical_rev_test_joint_z" not in joints


def test_convert_kinematic_tree():
    """Parent-child relationships are correct after conversion."""
    mjcf = _build_spherical_chain_mjcf()
    convert_spherical_joints_to_ball(mjcf)

    parent = mjcf.find(".//body[@name='parent_link']")
    children = [b.get("name") for b in parent.findall("body")]
    assert "child_link" in children
    assert "normal_child" in children
    assert "spherical_fake_test_joint_link1" not in children
    assert "spherical_fake_test_joint_link2" not in children


def test_convert_multiple_chains():
    """Two independent spherical chains under the same parent."""
    mjcf = ET.fromstring(
        """<mujoco>
  <worldbody>
    <body name="root" pos="0 0 0">
      <inertial mass="5.0" pos="0 0 0" diaginertia="1 1 1"/>
      <body name="fake_a1" pos="0.1 0 0">
        <joint name="rev_a_x" type="hinge" axis="1 0 0" damping="0"/>
        <body name="fake_a2">
          <joint name="rev_a_y" type="hinge" axis="0 1 0" damping="0"/>
          <body name="link_a">
            <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
            <joint name="rev_a_z" type="hinge" axis="0 0 1" damping="0"/>
            <geom type="sphere" size="0.05"/>
          </body>
        </body>
      </body>
      <body name="fake_b1" pos="0 0.2 0">
        <joint name="rev_b_x" type="hinge" axis="1 0 0" damping="0"/>
        <body name="fake_b2">
          <joint name="rev_b_y" type="hinge" axis="0 1 0" damping="0"/>
          <body name="link_b">
            <inertial mass="2.0" pos="0 0 0" diaginertia="0.2 0.2 0.2"/>
            <joint name="rev_b_z" type="hinge" axis="0 0 1" damping="0"/>
            <geom type="sphere" size="0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""
    )

    mass_before = _compute_total_mass(mjcf)
    convert_spherical_joints_to_ball(mjcf)
    mass_after = _compute_total_mass(mjcf)

    ball_joints = [j for j in mjcf.iter("joint") if j.get("type") == "ball"]
    assert len(ball_joints) == 2

    assert abs(mass_before - mass_after) < 1e-9

    body_names = _get_all_body_names(mjcf)
    for fake in ["fake_a1", "fake_a2", "fake_b1", "fake_b2"]:
        assert fake not in body_names

    root = mjcf.find(".//body[@name='root']")
    children = [b.get("name") for b in root.findall("body")]
    assert "link_a" in children
    assert "link_b" in children


def test_convert_preserves_nested_child_bodies():
    """Child bodies of the real link are preserved after conversion."""
    mjcf = ET.fromstring(
        """<mujoco>
  <worldbody>
    <body name="parent" pos="0 0 0">
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <body name="fake1" pos="0.1 0.2 0.3">
        <joint name="sph_x" type="hinge" axis="1 0 0" damping="0"/>
        <body name="fake2">
          <joint name="sph_y" type="hinge" axis="0 1 0" damping="0"/>
          <body name="child" pos="0 0 0">
            <inertial mass="0.5" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
            <joint name="sph_z" type="hinge" axis="0 0 1" damping="0"/>
            <geom type="sphere" size="0.05"/>
            <body name="grandchild" pos="0 0 -0.3">
              <inertial mass="0.3" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
              <joint name="gc_joint" type="hinge" axis="0 1 0"/>
              <geom type="sphere" size="0.03"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""
    )

    convert_spherical_joints_to_ball(mjcf)

    child = mjcf.find(".//body[@name='child']")
    assert child is not None
    grandchild = child.find("body[@name='grandchild']")
    assert grandchild is not None
    assert grandchild.find("joint").get("name") == "gc_joint"


def test_convert_transfers_quat():
    """Quaternion from the first fake body is transferred to the child."""
    mjcf = ET.fromstring(
        """<mujoco>
  <worldbody>
    <body name="parent" pos="0 0 0">
      <body name="fake1" pos="0.1 0 0" quat="0.707 0 0.707 0">
        <joint name="j_x" type="hinge" axis="1 0 0" damping="0"/>
        <body name="fake2">
          <joint name="j_y" type="hinge" axis="0 1 0" damping="0"/>
          <body name="child">
            <inertial mass="0.5" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
            <joint name="j_z" type="hinge" axis="0 0 1" damping="0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""
    )

    convert_spherical_joints_to_ball(mjcf)
    child = mjcf.find(".//body[@name='child']")
    assert child.get("pos") == "0.1 0 0"
    assert child.get("quat") == "0.707 0 0.707 0"


def test_convert_xml_roundtrip():
    """Converted MJCF can be serialized to XML string and re-parsed."""
    mjcf = _build_spherical_chain_mjcf()
    convert_spherical_joints_to_ball(mjcf)

    xml_str = ET.tostring(mjcf, encoding="unicode")
    reparsed = ET.fromstring(xml_str)

    # Same body count
    assert len(list(reparsed.iter("body"))) == len(list(mjcf.iter("body")))
    # Ball joint present
    ball = [j for j in reparsed.iter("joint") if j.get("type") == "ball"]
    assert len(ball) == 1


def test_convert_mujoco_default_hinge_type():
    """MuJoCo omits type='hinge' for default joints — conversion must still work."""
    mjcf = ET.fromstring(
        """<mujoco>
  <worldbody>
    <body name="parent" pos="0 0 0">
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <body name="fake1" pos="0.1 0.2 0.3">
        <inertial mass="1e-08" pos="0 0 0" diaginertia="1e-10 1e-10 1e-10"/>
        <joint name="sph_x" pos="0 0 0" axis="1 0 0"/>
        <body name="fake2">
          <inertial mass="1e-08" pos="0 0 0" diaginertia="1e-10 1e-10 1e-10"/>
          <joint name="sph_y" pos="0 0 0" axis="0 1 0"/>
          <body name="child" pos="0 0 0">
            <inertial mass="0.5" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
            <joint name="sph_z" pos="0 0 0" axis="0 0 1"/>
            <geom type="sphere" size="0.05"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""
    )
    mass_before = _compute_total_mass(mjcf)
    convert_spherical_joints_to_ball(mjcf)
    mass_after = _compute_total_mass(mjcf)

    ball = [j for j in mjcf.iter("joint") if j.get("type") == "ball"]
    assert len(ball) == 1

    body_names = _get_all_body_names(mjcf)
    assert "fake1" not in body_names
    assert "fake2" not in body_names
    assert "child" in body_names

    # Mass preserved (fake bodies had tiny mass that is dropped)
    assert abs(mass_before - mass_after) < 1e-4

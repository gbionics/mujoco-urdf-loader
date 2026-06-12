import xml.etree.ElementTree as ET

import pytest

from mujoco_urdf_loader.mjcf_fcn import (
    add_camera,
    add_equality_constraints_for_sites,
    add_joint_eq,
    add_joint_pos_sensor,
    add_joint_vel_sensor,
    add_new_worldbody,
    add_position_actuator,
    convert_hinge_to_ball_joints,
    separate_left_right_collision_groups,
    set_collision_groups,
    set_joint_damping,
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
# Tests for convert_hinge_to_ball_joints
# ---------------------------------------------------------------------------


def _make_mjcf_with_hinge():
    """Build a minimal MJCF with a hinge joint that should be converted to ball."""
    mjcf = ET.Element("mujoco")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body")
    body.set("name", "crank")

    child_body = ET.SubElement(body, "body")
    child_body.set("name", "rod")
    child_body.set("pos", "0.05 0.02 0")

    joint = ET.SubElement(child_body, "joint")
    joint.set("name", "spherical_rev_my_rod_x")
    joint.set("type", "hinge")
    joint.set("axis", "1 0 0")
    joint.set("range", "-6.28 6.28")
    joint.set("limited", "true")
    joint.set("damping", "0")
    joint.set("armature", "0")

    # Also add a regular hinge that should NOT be converted
    other_body = ET.SubElement(worldbody, "body")
    other_body.set("name", "other")
    other_joint = ET.SubElement(other_body, "joint")
    other_joint.set("name", "regular_hinge")
    other_joint.set("type", "hinge")
    other_joint.set("axis", "0 1 0")

    return mjcf


def test_convert_hinge_to_ball_joints():
    mjcf = _make_mjcf_with_hinge()
    ball_map = {"spherical_rev_my_rod_x": "my_rod"}

    convert_hinge_to_ball_joints(mjcf, ball_map)

    # The placeholder joint should now be a ball joint with new name
    ball = mjcf.find(".//joint[@name='my_rod_ball']")
    assert ball is not None
    assert ball.attrib["type"] == "ball"

    # Hinge-only attributes should be removed
    assert "axis" not in ball.attrib
    assert "ref" not in ball.attrib

    # Damping and armature should be set (defaults)
    assert float(ball.attrib["damping"]) == pytest.approx(0.01)
    assert float(ball.attrib["armature"]) == pytest.approx(0.001)

    # The old name should no longer exist
    assert mjcf.find(".//joint[@name='spherical_rev_my_rod_x']") is None


def test_convert_hinge_to_ball_custom_damping():
    mjcf = _make_mjcf_with_hinge()
    ball_map = {"spherical_rev_my_rod_x": "my_rod"}

    convert_hinge_to_ball_joints(mjcf, ball_map, damping=0.5, armature=0.05)

    ball = mjcf.find(".//joint[@name='my_rod_ball']")
    assert float(ball.attrib["damping"]) == pytest.approx(0.5)
    assert float(ball.attrib["armature"]) == pytest.approx(0.05)


def test_convert_hinge_to_ball_preserves_other_joints():
    mjcf = _make_mjcf_with_hinge()
    ball_map = {"spherical_rev_my_rod_x": "my_rod"}

    convert_hinge_to_ball_joints(mjcf, ball_map)

    # The regular hinge should be untouched
    regular = mjcf.find(".//joint[@name='regular_hinge']")
    assert regular is not None
    assert regular.attrib["type"] == "hinge"
    assert regular.attrib["axis"] == "0 1 0"


def test_convert_hinge_to_ball_missing_joint():
    mjcf = _make_mjcf_with_hinge()
    ball_map = {"nonexistent_joint": "some_base"}

    with pytest.raises(ValueError, match="nonexistent_joint"):
        convert_hinge_to_ball_joints(mjcf, ball_map)


def test_convert_hinge_to_ball_empty_map():
    mjcf = _make_mjcf_with_hinge()
    ball_map = {}

    # Should be a no-op
    convert_hinge_to_ball_joints(mjcf, ball_map)

    # Original joints should be unchanged
    assert mjcf.find(".//joint[@name='spherical_rev_my_rod_x']") is not None
    assert mjcf.find(".//joint[@name='regular_hinge']") is not None


# ---------------------------------------------------------------------------
# Tests for add_equality_constraints_for_sites
# ---------------------------------------------------------------------------


def _make_mjcf_with_sites():
    """Build a minimal MJCF with sites for equality constraint tests."""
    mjcf = ET.Element("mujoco")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body_a = ET.SubElement(worldbody, "body")
    body_a.set("name", "body_a")
    site_a = ET.SubElement(body_a, "site")
    site_a.set("name", "site_a")
    site_a.set("pos", "0 0 0")

    body_b = ET.SubElement(worldbody, "body")
    body_b.set("name", "body_b")
    site_b = ET.SubElement(body_b, "site")
    site_b.set("name", "site_b")
    site_b.set("pos", "0.1 0.2 0.3")

    site_c = ET.SubElement(body_b, "site")
    site_c.set("name", "site_c")
    site_c.set("pos", "0.4 0.5 0.6")

    return mjcf


def test_add_equality_constraints_connect():
    mjcf = _make_mjcf_with_sites()

    mjcf = add_equality_constraints_for_sites(mjcf, [("site_a", "site_b")])

    assert len(mjcf.findall(".//equality")) == 1
    connects = mjcf.findall(".//equality/connect")
    assert len(connects) == 1
    assert connects[0].attrib["site1"] == "site_a"
    assert connects[0].attrib["site2"] == "site_b"


def test_add_equality_constraints_multiple():
    mjcf = _make_mjcf_with_sites()

    mjcf = add_equality_constraints_for_sites(
        mjcf, [("site_a", "site_b"), ("site_b", "site_c")]
    )

    # Only one <equality> element should exist
    assert len(mjcf.findall(".//equality")) == 1
    connects = mjcf.findall(".//equality/connect")
    assert len(connects) == 2
    assert connects[0].attrib["site1"] == "site_a"
    assert connects[1].attrib["site1"] == "site_b"
    assert connects[1].attrib["site2"] == "site_c"


def test_add_equality_constraints_weld():
    mjcf = _make_mjcf_with_sites()

    mjcf = add_equality_constraints_for_sites(
        mjcf, [("site_a", "site_b")], constraint_type="weld"
    )

    welds = mjcf.findall(".//equality/weld")
    assert len(welds) == 1
    assert welds[0].attrib["body1"] == "body_a"
    assert welds[0].attrib["body2"] == "body_b"


def test_add_equality_constraints_missing_site():
    mjcf = _make_mjcf_with_sites()

    with pytest.raises(ValueError, match="nonexistent"):
        add_equality_constraints_for_sites(mjcf, [("site_a", "nonexistent")])


def test_add_equality_constraints_unknown_type():
    mjcf = _make_mjcf_with_sites()

    with pytest.raises(ValueError, match="Unknown constraint type"):
        add_equality_constraints_for_sites(
            mjcf, [("site_a", "site_b")], constraint_type="invalid"
        )

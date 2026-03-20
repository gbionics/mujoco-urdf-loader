import xml.etree.ElementTree as ET
from typing import Dict, List
from scipy.spatial.transform import Rotation


def add_new_worldbody(
    mjcf: ET.Element,
    x: float = 0,
    y: float = 0,
    z: float = 0,
    r_x: float = 0,
    r_y: float = 0,
    r_z: float = 0,
    freeze_root: bool = True,
) -> ET.Element:
    """Add a new worldbody to the mjcf file with the robot as a child and remove the old worldbody.

    Args:
        mjcf (ET.Element): The mjcf file.
        x (float): The x position of the robot (m).
        y (float): The y position of the robot (m).
        z (float): The z position of the robot (m).
        r_x (float): The x rotation of the robot (rad).
        r_y (float): The y rotation of the robot (rad).
        r_z (float): The z rotation of the robot (rad).
    """

    robot = ET.Element("body")
    robot.set("name", "robot_base")
    robot.set("pos", f"{x} {y} {z}")
    robot.set("euler", f"{r_x} {r_y} {r_z}")
    for child in mjcf.find("worldbody"):
        robot.append(child)
    mjcf.remove(mjcf.find("worldbody"))

    worldbody = ET.SubElement(mjcf, "worldbody")
    worldbody.append(robot)

    if not freeze_root:
        ET.SubElement(robot, "freejoint", {"name": "base_joint"})

    return mjcf


def add_position_actuator(
    mjcf: ET.Element,
    joint: str,
    ctrlrange: List[float] = None,
    kp: float = 10,
    group: int = 0,
    name: str = None,
) -> ET.Element:
    """Add a position actuator to the joint.

    Args:
        mjcf (ET.Element): The mjcf file.
        joint (str): The joint to add the actuator to.
        ctrlrange (List[float]): The control range of the actuator (default: -1, 1).
        kp (float): The proportional gain of the actuator (default: 10).
        group (int): The group of the actuator (default: 0).
        name (str): The name of the actuator (default: f"{joint}_motor").
    """

    # check if the ctrlrange is None
    if ctrlrange is None:
        ctrlrange = [-1, 1]

    # check if there already is an actuator element in the mjcf
    if mjcf.find(".//actuator") is None:
        actuators = ET.Element("actuator")
        mjcf.append(actuators)
    else:
        actuators = mjcf.find(".//actuator")

    # check if there is already an actuator element for the joint
    for actuator in actuators:
        if actuator.attrib["joint"] == joint:
            return mjcf

    # create the position actuator
    motor = ET.SubElement(actuators, "position")
    motor.set("joint", joint)
    motor.set("name", name if name is not None else f"{joint}_motor")
    motor.set("ctrlrange", f"{ctrlrange[0]} {ctrlrange[1]}")
    motor.set("kp", str(kp))
    motor.set("group", str(group))

    return mjcf


def add_torque_actuator(
    mjcf: ET.Element,
    joint: str,
    ctrlrange: List[float] = None,
    name: str = None,
) -> ET.Element:
    """Add a torque actuator to the joint.

    Args:
        mjcf (ET.Element): The mjcf file.
        joint (str): The joint to add the actuator to.
        ctrlrange (List[float]): The control range of the actuator (default: -1, 1).
        kp (float): The proportional gain of the actuator (default: 10).
        group (int): The group of the actuator (default: 0).
        name (str): The name of the actuator (default: f"{joint}_motor").
    """

    # check if the ctrlrange is None
    if ctrlrange is None:
        ctrlrange = [-1000, 1000]

    # check if there already is an actuator element in the mjcf
    if mjcf.find(".//actuator") is None:
        actuators = ET.Element("actuator")
        mjcf.append(actuators)
    else:
        actuators = mjcf.find(".//actuator")

    # check if there is already an actuator element for the joint
    for actuator in actuators:
        if actuator.attrib["joint"] == joint:
            return mjcf

    # create the torque actuator
    motor = ET.SubElement(actuators, "motor")
    motor.set("name", name if name is not None else f"{joint}")
    motor.set("joint", joint)
    motor.set("ctrlrange", f"{ctrlrange[0]} {ctrlrange[1]}")
    motor.set("gear", "1")
    return mjcf


def add_joint_pos_sensor(mjcf: ET.Element, joint: str, name: str = None) -> ET.Element:
    """Add a joint position sensor to the joint.

    Args:
        mjcf (ET.Element): The mjcf file.
        joint (str): The joint to add the sensor to.
        name (str): The name of the sensor (default: f"{joint}_pos").
    """

    # check if there already is a sensor element in the mjcf
    if mjcf.find(".//sensor") is None:
        sensors = ET.Element("sensor")
        mjcf.append(sensors)
    else:
        sensors = mjcf.find(".//sensor")

    # create the joint position sensor
    pos = ET.SubElement(sensors, "jointpos")
    pos.set("joint", joint)
    pos.set("name", name if name is not None else f"{joint}_pos")

    return mjcf


def add_joint_vel_sensor(mjcf: ET.Element, joint: str, name: str = None) -> ET.Element:
    """Add a joint velocity sensor to the joint.

    Args:
        mjcf (ET.Element): The mjcf file.
        joint (str): The joint to add the sensor to.
        name (str): The name of the sensor (default: f"{joint}_vel").
    """

    # check if there already is a sensor element in the mjcf
    if mjcf.find(".//sensor") is None:
        sensors = ET.Element("sensor")
        mjcf.append(sensors)
    else:
        sensors = mjcf.find(".//sensor")

    # create the joint velocity sensor
    vel = ET.SubElement(sensors, "jointvel")
    vel.set("joint", joint)
    vel.set("name", name if name is not None else f"{joint}_vel")

    return mjcf

def add_gyro_sensor(mjcf: ET.Element, site: str, name: str = None) -> ET.Element:
    """Add a gyroscope sensor to the body.

    Args:
        mjcf (ET.Element): The mjcf file.
        site (str): The site to add the sensor to.
        name (str): The name of the sensor (default: f"{site}_gyro").
    """

    # check if there already is a sensor element in the mjcf
    if mjcf.find(".//sensor") is None:
        sensors = ET.Element("sensor")
        mjcf.append(sensors)
    else:
        sensors = mjcf.find(".//sensor")

    # create the gyroscope sensor
    gyro = ET.SubElement(sensors, "gyro")
    gyro.set("name", name if name is not None else f"{site}_gyro")
    gyro.set("site", site)

    return mjcf

def add_camera_to_site(mjcf: ET.Element, name: str, site: str, fovy: float) -> ET.Element:
    """Add a camera tag to the site.

    Args:
        mjcf (ET.Element): The mjcf file.
        name (str): The name of the camera.
        site (str): The site to add the camera tag to.
        fovy (float): The field of view of the camera (deg).
    """
    # look for the site in the mjcf file and retrieve the position and orientation of the site
    site_element = mjcf.find(f".//site[@name='{site}']")
    if site_element is None:
        raise ValueError(f"Site {site} not found in the mjcf file.")
    pos = site_element.attrib["pos"]
    quat = site_element.attrib["quat"]

    q_site_wxyz = list(map(float, quat.split()))
    if len(q_site_wxyz) != 4:
        raise ValueError("Quaternion must have exactly 4 components in [w x y z] format.")

    # scipy uses [x, y, z, w], MuJoCo uses [w, x, y, z]
    r_site = Rotation.from_quat([q_site_wxyz[1], q_site_wxyz[2], q_site_wxyz[3], q_site_wxyz[0]])
    # MuJoCo cameras look along the local -Z axis, while site/tool frames are
    # typically defined with +Z as forward. Rotate 180° about X so the camera
    # optical axis aligns with the intended site forward direction.
    r_x180 = Rotation.from_euler("x", 180, degrees=True)
    r_camera = r_site * r_x180
    q_camera_xyzw = r_camera.as_quat()
    camera_quat = f"{q_camera_xyzw[3]} {q_camera_xyzw[0]} {q_camera_xyzw[1]} {q_camera_xyzw[2]}"

    # create the camera tag as a sibling right after the site element
    camera = ET.Element("camera")
    camera.set("name", name)
    camera.set("pos", pos)
    camera.set("quat", camera_quat)
    camera.set("fovy", str(fovy))

    # xml.etree.ElementTree.Element has no addnext(); find parent and insert manually
    parent = None
    for elem in mjcf.iter():
        for child in list(elem):
            if child is site_element:
                parent = elem
                break
        if parent is not None:
            break

    if parent is None:
        raise ValueError(f"Parent of site {site} not found in the mjcf file.")

    children = list(parent)
    site_idx = children.index(site_element)
    parent.insert(site_idx + 1, camera)
    return mjcf


def add_framequat_sensor(mjcf: ET.Element, objname: str, objtype: str = 'site', name: str = None) -> ET.Element:
    """Add a frame quaternion sensor to the body.

    Args:
        mjcf (ET.Element): The mjcf file.
        site (str): The site to add the sensor to.
        name (str): The name of the sensor (default: f"{site}_framequat").
    """

    # check if there already is a sensor element in the mjcf
    if mjcf.find(".//sensor") is None:
        sensors = ET.Element("sensor")
        mjcf.append(sensors)
    else:
        sensors = mjcf.find(".//sensor")

    # create the frame quaternion sensor
    framequat = ET.SubElement(sensors, "framequat")
    framequat.set("name", name if name is not None else f"{objname}_framequat")
    framequat.set("objtype", objtype)
    framequat.set("objname", objname)

    return mjcf


def add_joint_eq(
    mjcf: ET.Element,
    joint1: str,
    joint2: str,
    name: str = None,
    multiplier: float = 1.0,
    offset: float = 0.0,
) -> ET.Element:
    """Add a joint equality constraint between two joints.

    Args:
        mjcf (ET.Element): The mjcf file.
        joint1 (str): The first joint, the dependent joint.
        joint2 (str): The second joint, the independent joint.
        name (str): The name of the equality constraint (default: f"{joint1}_{joint2}").
        multiplier (float): The multiplier for the equality constraint (default: 1.0).
        offset (float): The offset for the equality constraint (default: 0.0).
    """

    # check if there already is an equality element in the mjcf
    if mjcf.find(".//equality") is None:
        equality = ET.Element("equality")
        mjcf.append(equality)
    else:
        equality = mjcf.find(".//equality")

    # create the joint equality constraint
    dist_eq = ET.SubElement(equality, "joint")
    dist_eq.set("name", name if name is not None else f"{joint1}_{joint2}")
    dist_eq.set("joint1", joint1)
    dist_eq.set("joint2", joint2)

    # Set the polynomial coefficients for the equality constraint: joint1 = multiplier * joint2 + offset
    polycoef = f"{offset} {multiplier} 0 0 0"
    dist_eq.set("polycoef", polycoef)

    return mjcf


def add_camera(
    body: ET.Element,
    name: str = "camera",
    x: float = 0,
    y: float = 0,
    z: float = 0,
    r_x: float = 0,
    r_y: float = 0,
    r_z: float = 0,
) -> ET.Element:
    """Add a camera to the body.

    Args:
        body (ET.Element): The body to add the camera to.
        name (str): The name of the camera (default: "camera").
        x (float): The x position of the camera (m).
        y (float): The y position of the camera (m).
        z (float): The z position of the camera (m).
        r_x (float): The x rotation of the camera (rad).
        r_y (float): The y rotation of the camera (rad).
        r_z (float): The z rotation of the camera (rad).
    """

    camera = ET.SubElement(body, "camera")
    camera.set("name", name)
    camera.set("pos", f"{x} {y} {z}")
    camera.set("euler", f"{r_x} {r_y} {r_z}")

    return body


def set_collision_groups(
    mjcf: ET.Element, idx: str = "", group: int = 0b000, affinity: int = 0b000
) -> ET.Element:
    """Set the collision groups and affinities in the mjcf file.

    Args:
        mjcf (ET.Element): The mjcf file.
        idx (str): The index of the collision group.
        group (int): The collision group (default: 0b000).
        affinity (int): The collision affinity (default: 0b000).
    """

    for geom in mjcf.findall(".//geom"):
        if "mesh" not in geom.attrib:
            continue
        if idx in geom.attrib["mesh"]:
            geom.set("contype", str(group))
            geom.set("conaffinity", str(affinity))

    return mjcf

def update_meshdir(mjcf: ET.Element, meshdir: str) -> ET.Element:
    """Set the meshdir in the mjcf file.

    Args:
        mjcf (ET.Element): The mjcf file.
        meshdir (str): The mesh directory to set in the mjcf file.
    """

    compiler = mjcf.find(".//compiler")
    if compiler is None:
        compiler = ET.SubElement(mjcf, "compiler")
    compiler.set("meshdir", meshdir)

    return mjcf

def separate_left_right_collision_groups(
    mjcf: ET.Element,
    l_group: int = 0b001,
    l_aff: int = 0b110,
    r_group: int = 0b100,
    r_aff: int = 0b011,
    root_group: int = 0b000,
    root_aff: int = 0b000,
    def_group: int = 0b010,
    def_aff: int = 0b101,
) -> ET.Element:
    """Separate the left and right collision groups in the mjcf file.

    Args:
        mjcf (ET.Element): The mjcf file.
        l_group (int): The left collision group (default: 0b001).
        l_aff (int): The left collision affinity (default: 0b110).
        r_group (int): The right collision group (default: 0b100).
        r_aff (int): The right collision affinity (default: 0b011).
        root_group (int): The root collision group (default: 0b000).
        root_aff (int): The root collision affinity (default: 0b000).
        def_group (int): The default collision group (default: 0b010).
        def_aff (int): The default collision affinity (default: 0b101).
    """

    set_collision_groups(mjcf, idx="", group=def_group, affinity=def_aff)
    set_collision_groups(mjcf, idx="_l_", group=l_group, affinity=l_aff)
    set_collision_groups(mjcf, idx="_r_", group=r_group, affinity=r_aff)
    set_collision_groups(mjcf, idx="root", group=root_group, affinity=root_aff)

    return mjcf


def set_joint_damping(
    mjcf: ET.Element, subset: List[str] = None, damping: float = 0.005
) -> ET.Element:
    """Set the damping of the joints in the mjcf file.

    Args:
        mjcf (ET.Element): The mjcf file.
        subset (List[str]): The subset of joints to set the damping for.
        damping (float): The damping of the joints (default: 0.005).
    """

    if subset is not None:
        for joint in mjcf.findall(".//body/joint"):
            if any(joint_element in joint.attrib["name"] for joint_element in subset):
                joint.set("damping", str(damping))
    else:
        for joint in mjcf.findall(".//body/joint"):
            joint.set("damping", str(damping))

    return mjcf


def add_box(
    mjcf: ET.Element,
    name: str,
    pos: List[float] = [0, 0, 0],
    size: List[float] = [0.1, 0.1, 0.1],
    rgba: List[float] = [1, 0, 0, 1],
    mass: float = 1,
) -> ET.Element:
    """Add a box to the mjcf file.

    Args:
        mjcf (ET.Element): The mjcf file.
        name (str): The name of the box.
        type (str): The type of the box.
        pos (List[float]): The position of the box.
        size (List[float]): The size of the box.
        rgba (List[float]): The color of the box.
        mass (float): The mass of the box.
    """

    # find the worldbody element, if it does not exist create it
    if mjcf.find(".//worldbody") is None:
        worldbody = ET.SubElement(mjcf, "worldbody")
    else:
        worldbody = mjcf.find(".//worldbody")

    # create the body element
    body = ET.SubElement(worldbody, "body")
    body.set("name", name)
    body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")

    # create the freejoint element
    ET.SubElement(body, "freejoint")

    # create the geom element
    geom = ET.SubElement(body, "geom")
    geom.set("name", f"{name}_geom")
    geom.set("type", "box")
    geom.set("size", f"{size[0]} {size[1]} {size[2]}")
    geom.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
    geom.set("mass", f"{mass}")

    return mjcf


def add_sphere(
    mjcf: ET.Element,
    name: str,
    pos: List[float] = [0, 0, 0],
    size: float = 0.1,
    rgba: List[float] = [1, 0, 0, 1],
    mass: float = 1,
) -> ET.Element:
    """Add a sphere to the mjcf file.

    Args:
        mjcf (ET.Element): The mjcf file.
        name (str): The name of the sphere.
        type (str): The type of the sphere.
        pos (List[float]): The position of the sphere.
        size (float): The size of the sphere.
        rgba (List[float]): The color of the sphere.
        mass (float): The mass of the sphere.
    """

    # find the worldbody element, if it does not exist create it
    if mjcf.find(".//worldbody") is None:
        worldbody = ET.SubElement(mjcf, "worldbody")
    else:
        worldbody = mjcf.find(".//worldbody")

    # create the body element
    body = ET.SubElement(worldbody, "body")
    body.set("name", name)
    body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")

    # create the freejoint element
    ET.SubElement(body, "freejoint")

    # create the geom element
    geom = ET.SubElement(body, "geom")
    geom.set("name", f"{name}_geom")
    geom.set("type", "sphere")
    geom.set("size", f"{size}")
    geom.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")
    geom.set("mass", f"{mass}")

    return mjcf


def convert_hinge_to_ball_joints(
    mjcf: ET.Element,
    ball_joint_map: Dict[str, str],
    damping: float = 0.01,
    armature: float = 0.001,
    frictionloss: float = 0.001,
) -> ET.Element:
    """Convert placeholder hinge joints into MuJoCo ball joints.

    After collapsing 3-revolute spherical triplets in the URDF, MuJoCo creates
    regular hinge joints for them.  This function post-processes the MJCF to
    turn those hinges into ``type="ball"`` joints, removes hinge-only
    attributes, and sets damping / armature for numerical stability.

    Args:
        mjcf: The MJCF root element (modified in-place).
        ball_joint_map: Mapping from the placeholder joint name (the former
            ``_x`` revolute joint name) to the spherical group base name.
            Produced by
            :func:`~mujoco_urdf_loader.urdf_fcn.collapse_spherical_revolute_triplets`.
        damping: Viscous damping coefficient applied to each ball joint.
            A small positive value prevents unconstrained free-spin that
            causes simulation instability.  Default ``0.01``.
        armature: Rotor inertia (diagonal) added to each ball joint.
            Default ``0.001``.
        frictionloss: Friction loss applied to each ball joint.
            Default ``0.001``.

    Returns:
        The modified MJCF element.

    Raises:
        ValueError: If a placeholder joint name is not found in the MJCF.
    """
    for placeholder_name, base_name in ball_joint_map.items():
        joint_elem = mjcf.find(f".//joint[@name='{placeholder_name}']")
        if joint_elem is None:
            raise ValueError(
                f"Placeholder joint '{placeholder_name}' for spherical group "
                f"'{base_name}' not found in the MJCF model."
            )

        # Rename and switch type
        joint_elem.set("name", f"{base_name}_ball")
        joint_elem.set("type", "ball")

        # Remove attributes that are only meaningful for hinge / slide joints.
        # ``limited`` / ``range`` from revolute joints are invalid for ball
        # joints (ball range[0] must be 0 and represents max deflection).
        for attr in ("axis", "ref", "limited", "range"):
            if attr in joint_elem.attrib:
                del joint_elem.attrib[attr]

        # Set damping and armature for numerical stability
        joint_elem.set("damping", str(damping))
        joint_elem.set("armature", str(armature))
        joint_elem.set("frictionloss", str(frictionloss))
    return mjcf

def add_equality_constraints_for_sites(
    mjcf: ET.Element, site_pairs: List[tuple], constraint_type: str = "connect"
) -> ET.Element:
    """
    Add equality constraints between pairs of sites in MJCF.

    Args:
        mjcf (ET.Element): The MJCF file as ElementTree.
        site_pairs (List[tuple]): List of tuples with (site1_name, site2_name) to connect.
        constraint_type (str): Type of constraint - "connect" or "weld" (default: "connect").

    Returns:
        ET.Element: The modified MJCF file.
    """
    # Find or create the equality element
    equality = mjcf.find("equality")
    if equality is None:
        equality = ET.SubElement(mjcf, "equality")

    for site1, site2 in site_pairs:
        # Verify both sites exist
        site1_elem = mjcf.find(f".//site[@name='{site1}']")
        site2_elem = mjcf.find(f".//site[@name='{site2}']")

        if site1_elem is None:
            raise ValueError(f"Site {site1} not found in MJCF")
        if site2_elem is None:
            raise ValueError(f"Site {site2} not found in MJCF")

        # Create the equality constraint
        if constraint_type == "connect":
            # Connect constraint directly references sites (no anchor needed for sites)
            constraint = ET.SubElement(equality, "connect")
            constraint.set("site1", site1)
            constraint.set("site2", site2)
        elif constraint_type == "weld":
            # Weld constraint references bodies
            # Find parent bodies of the sites
            body1 = None
            body2 = None
            for body in mjcf.findall(".//body"):
                if body.find(f".//site[@name='{site1}']") is not None:
                    body1 = body.attrib.get("name")
                if body.find(f".//site[@name='{site2}']") is not None:
                    body2 = body.attrib.get("name")

            if body1 is None or body2 is None:
                raise ValueError(
                    f"Could not find parent bodies for sites {site1} and {site2}"
                )

            constraint = ET.SubElement(equality, "weld")
            constraint.set("body1", body1)
            constraint.set("body2", body2)
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")

        print(
            f"Created {constraint_type} equality constraint between {site1} and {site2}"
        )

    return mjcf

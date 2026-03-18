import xml.etree.ElementTree as ET
from typing import List


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


def add_sites_for_fixed_joints(mjcf: ET.Element, urdf: ET.Element) -> ET.Element:
    """
    Add sites to all fixed joints in their parent body.

    Args:
        mjcf (ET.Element): The MJCF file as ElementTree.
        urdf (ET.Element): The URDF file as ElementTree.

    Returns:
        ET.Element: The modified MJCF file.
    """
    for fixed_joint in urdf.findall(".//joint[@type='fixed']"):
        parent_link = fixed_joint.find("parent").attrib["link"]
        joint_name = fixed_joint.attrib["name"]

        body = mjcf.find(f".//body[@name='{parent_link}']")
        if body is None:
            print(f"Body {parent_link} not found in mjcf")
            continue

        origin = fixed_joint.find("origin")
        if origin is None:
            continue

        xyz = origin.attrib.get("xyz", "0 0 0")
        rpy = list(map(float, origin.attrib.get("rpy", "0 0 0").split()))

        rotation = idyntree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
        quaternion = rotation.asQuaternion()

        site = ET.SubElement(body, "site")
        site.set("name", joint_name)
        site.set("pos", xyz)
        site.set(
            "quat", f"{quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]}"
        )

    return mjcf


def _idyn_rotation_to_numpy(idyn_rot):
    """Convert iDynTree Rotation to 3x3 numpy array."""
    R = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            R[i, j] = idyn_rot.getVal(i, j)
    return R


def _quat_str_to_rotation_matrix(quat_str):
    """Convert quaternion string (w x y z) to 3x3 rotation matrix."""
    w, x, y, z = map(float, quat_str.split())
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _parallel_axis_term(mass, d):
    """Compute the parallel axis theorem offset: mass * (|d|^2 I - d d^T)."""
    d = np.asarray(d, dtype=float)
    return mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


def _delump_parent_inertial(parent_body, urdf_child_link, joint_xyz, joint_rot_matrix):
    """
    Subtract the child link's lumped inertial contribution from the parent body.

    When MuJoCo compiles a URDF with fixed joints, child masses are lumped into
    the parent. This function reverses that for a single child so the parent
    and child each carry their own correct mass properties.
    """
    parent_inertial = parent_body.find("inertial")
    if parent_inertial is None or urdf_child_link is None:
        return
    child_inertial_elem = urdf_child_link.find("inertial")
    if child_inertial_elem is None:
        return

    # --- Lumped parent properties (from MJCF) ---
    M = float(parent_inertial.get("mass", "0"))

    mass_elem = child_inertial_elem.find("mass")
    m_c = float(mass_elem.get("value", "0")) if mass_elem is not None else 0.0
    if m_c <= 0 or m_c >= M:
        return

    r_lumped = np.array(list(map(float, parent_inertial.get("pos", "0 0 0").split())))

    # Reconstruct lumped inertia tensor in body frame
    R_inertial = _quat_str_to_rotation_matrix(parent_inertial.get("quat", "1 0 0 0"))
    if parent_inertial.get("fullinertia") is not None:
        fi = list(map(float, parent_inertial.get("fullinertia").split()))
        I_diag_frame = np.array(
            [[fi[0], fi[3], fi[4]], [fi[3], fi[1], fi[5]], [fi[4], fi[5], fi[2]]]
        )
    elif parent_inertial.get("diaginertia") is not None:
        di = list(map(float, parent_inertial.get("diaginertia").split()))
        I_diag_frame = np.diag(di)
    else:
        return
    I_lumped = R_inertial @ I_diag_frame @ R_inertial.T

    # --- Child inertial in parent body frame ---
    child_origin = child_inertial_elem.find("origin")
    r_c_local = np.zeros(3)
    R_c_origin = np.eye(3)
    if child_origin is not None:
        r_c_local = np.array(list(map(float, child_origin.get("xyz", "0 0 0").split())))
        c_rpy = list(map(float, child_origin.get("rpy", "0 0 0").split()))
        R_c_origin = _idyn_rotation_to_numpy(
            idyntree.Rotation.RPY(c_rpy[0], c_rpy[1], c_rpy[2])
        )

    inertia_elem = child_inertial_elem.find("inertia")
    if inertia_elem is None:
        return

    I_c_local = np.array(
        [
            [
                float(inertia_elem.get("ixx", "0")),
                float(inertia_elem.get("ixy", "0")),
                float(inertia_elem.get("ixz", "0")),
            ],
            [
                float(inertia_elem.get("ixy", "0")),
                float(inertia_elem.get("iyy", "0")),
                float(inertia_elem.get("iyz", "0")),
            ],
            [
                float(inertia_elem.get("ixz", "0")),
                float(inertia_elem.get("iyz", "0")),
                float(inertia_elem.get("izz", "0")),
            ],
        ]
    )
    # Rotate inertia from child-inertial frame to child-link frame
    I_c_link = R_c_origin @ I_c_local @ R_c_origin.T

    # Transform to parent body frame
    p_joint = np.asarray(joint_xyz, dtype=float)
    R_joint = np.asarray(joint_rot_matrix, dtype=float)
    r_c = R_joint @ r_c_local + p_joint
    I_c_parent = R_joint @ I_c_link @ R_joint.T

    # --- De-lump ---
    m_p = M - m_c
    r_p = (M * r_lumped - m_c * r_c) / m_p

    # Child inertia shifted to lumped CoM
    I_c_at_lumped = I_c_parent + _parallel_axis_term(m_c, r_lumped - r_c)
    # Parent inertia at lumped CoM
    I_p_at_lumped = I_lumped - I_c_at_lumped
    # Shift parent inertia to new parent CoM
    I_p = I_p_at_lumped - _parallel_axis_term(m_p, r_lumped - r_p)

    # --- Write back ---
    parent_inertial.set("mass", str(m_p))
    parent_inertial.set("pos", f"{r_p[0]} {r_p[1]} {r_p[2]}")
    parent_inertial.set("quat", "1 0 0 0")
    if "diaginertia" in parent_inertial.attrib:
        del parent_inertial.attrib["diaginertia"]
    parent_inertial.set(
        "fullinertia",
        f"{I_p[0,0]} {I_p[1,1]} {I_p[2,2]} {I_p[0,1]} {I_p[0,2]} {I_p[1,2]}",
    )


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
            print(f"Warning: Site {site1} not found in MJCF")
            continue
        if site2_elem is None:
            print(f"Warning: Site {site2} not found in MJCF")
            continue

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
                print(
                    f"Warning: Could not find parent bodies for sites {site1} and {site2}"
                )
                continue

            constraint = ET.SubElement(equality, "weld")
            constraint.set("body1", body1)
            constraint.set("body2", body2)
        else:
            print(f"Unknown constraint type: {constraint_type}")
            continue

        print(
            f"Created {constraint_type} equality constraint between {site1} and {site2}"
        )

    return mjcf


def add_sensors_to_sites(mjcf: ET.Element) -> ET.Element:
    """
    Add force-torque and IMU sensors to the specified sites in the MJCF file.

    Args:
        mjcf (ET.Element): The MJCF file as ElementTree.

    Returns:
        ET.Element: The modified MJCF file.
    """
    # Ensure the sensors element exists
    if mjcf.find(".//sensor") is None:
        sensors = ET.Element("sensor")
        mjcf.append(sensors)
    else:
        sensors = mjcf.find(".//sensor")

    # Find all sites with "fts" or "imu" in their names
    sites = mjcf.findall(".//site")
    for site in sites:
        site_name = site.get("name")
        if "_ft_" in site_name:
            add_force_torque_sensors(sensors, site_name)
        elif "_imu" in site_name:
            add_imu_sensors(sensors, site_name)

    return mjcf


def add_force_torque_sensors(sensors: ET.Element, site_name: str) -> None:
    """
    Add force and torque sensors to a specific site.

    Args:
        sensors (ET.Element): The sensor element in the MJCF file.
        site_name (str): The name of the site.
    """
    force_sensor = ET.SubElement(sensors, "force")
    force_sensor.set("name", f"{site_name}_force_sensor")
    force_sensor.set("site", site_name)

    torque_sensor = ET.SubElement(sensors, "torque")
    torque_sensor.set("name", f"{site_name}_torque_sensor")
    torque_sensor.set("site", site_name)


def add_imu_sensors(sensors: ET.Element, site_name: str) -> None:
    """
    Add accelerometer and gyroscope sensors to a specific site.

    Args:
        sensors (ET.Element): The sensor element in the MJCF file.
        site_name (str): The name of the site.
    """
    accelerometer_sensor = ET.SubElement(sensors, "accelerometer")
    accelerometer_sensor.set("name", f"{site_name}_acc_sensor")
    accelerometer_sensor.set("site", site_name)

    gyroscope_sensor = ET.SubElement(sensors, "gyro")
    gyroscope_sensor.set("name", f"{site_name}_gyro_sensor")
    gyroscope_sensor.set("site", site_name)


# ---------------------------------------------------------------------------
# Spherical joint detection & conversion (3 revolute -> 1 ball)
# ---------------------------------------------------------------------------


def _is_zero_mass_body(body: ET.Element, tolerance: float = 1e-6) -> bool:
    """Check if a MJCF body has zero or negligible mass."""
    inertial = body.find("inertial")
    if inertial is None:
        return True
    mass = float(inertial.get("mass", "0"))
    return abs(mass) < tolerance


def _get_hinge_joints(body: ET.Element) -> List[ET.Element]:
    """Get all direct hinge joint children of a body.

    MuJoCo's default joint type is ``hinge`` and it omits the ``type``
    attribute when saving XML. So a joint is hinge if ``type`` is either
    ``"hinge"`` or absent (not ``ball``, ``slide``, or ``free``).
    """
    _NON_HINGE = {"ball", "slide", "free"}
    return [
        j for j in body.findall("joint") if j.get("type", "hinge") not in _NON_HINGE
    ]


def _derive_ball_joint_name(first_joint_name: str) -> str:
    """Derive ball joint name from the first revolute joint in the chain.

    Following iDynTree convention: strip the ``_x`` suffix from the first
    joint name.  Falls back to stripping from ``_rev_`` if ``_x`` suffix is
    not found.
    """
    if first_joint_name.endswith("_x"):
        return first_joint_name[:-2]
    idx = first_joint_name.rfind("_rev_")
    if idx >= 0:
        return first_joint_name[:idx]
    return first_joint_name + "_ball"


def find_spherical_joint_patterns(
    mjcf: ET.Element, zero_mass_tolerance: float = 1e-4
) -> List[dict]:
    """Detect 3-hinge chains with 2 zero-mass intermediate bodies in MJCF.

    Finds patterns produced by converting fixed joints to 3 revolute joints
    (iDynTree / convert_fixed_to_spherical convention)::

        parent -> fake1 (zero mass, 1 hinge, no geoms) ->
                  fake2 (zero mass, 1 hinge, no geoms) ->
                  child (has hinge joint)

    Returns:
        List of dicts with keys: ``parent_body``, ``first_fake``,
        ``second_fake``, ``real_body``, ``joint1``, ``joint2``, ``joint3``.
    """
    patterns = []

    for parent_body in mjcf.iter("body"):
        for first_fake in list(parent_body.findall("body")):
            # --- first intermediate body ---
            if not _is_zero_mass_body(first_fake, zero_mass_tolerance):
                continue
            first_hinges = _get_hinge_joints(first_fake)
            if len(first_hinges) != 1:
                continue
            if first_fake.findall("geom"):
                continue
            first_children = first_fake.findall("body")
            if len(first_children) != 1:
                continue

            # --- second intermediate body ---
            second_fake = first_children[0]
            if not _is_zero_mass_body(second_fake, zero_mass_tolerance):
                continue
            second_hinges = _get_hinge_joints(second_fake)
            if len(second_hinges) != 1:
                continue
            if second_fake.findall("geom"):
                continue
            second_children = second_fake.findall("body")
            if len(second_children) != 1:
                continue

            # --- real child body (must have at least one hinge) ---
            real_body = second_children[0]
            real_hinges = _get_hinge_joints(real_body)
            if len(real_hinges) < 1:
                continue

            patterns.append(
                {
                    "parent_body": parent_body,
                    "first_fake": first_fake,
                    "second_fake": second_fake,
                    "real_body": real_body,
                    "joint1": first_hinges[0],
                    "joint2": second_hinges[0],
                    "joint3": real_hinges[0],
                }
            )

    return patterns


def convert_spherical_joints_to_ball(
    mjcf: ET.Element, zero_mass_tolerance: float = 1e-4
) -> ET.Element:
    """Replace 3-hinge chains with ball joints in MJCF.

    Detects chains of 3 hinge joints with 2 zero-mass intermediate bodies
    (produced by converting URDF fixed joints to spherical via 3 revolute
    joints) and collapses each chain into a single MuJoCo ball joint.

    The combined transform of the chain is carried by the first intermediate
    body's position/orientation, which is transferred to the real child body.

    Args:
        mjcf: Root MJCF element tree.
        zero_mass_tolerance: Mass threshold for detecting zero-mass bodies.

    Returns:
        The modified MJCF element tree.
    """
    patterns = find_spherical_joint_patterns(mjcf, zero_mass_tolerance)

    for pattern in patterns:
        parent_body = pattern["parent_body"]
        first_fake = pattern["first_fake"]
        second_fake = pattern["second_fake"]
        real_body = pattern["real_body"]
        joint1 = pattern["joint1"]
        joint3 = pattern["joint3"]

        # Derive ball joint name (strip '_x' suffix per iDynTree convention)
        ball_name = _derive_ball_joint_name(joint1.get("name", "ball"))

        # Transfer position/orientation from first_fake to real_body
        real_body.set("pos", first_fake.get("pos", "0 0 0"))
        first_quat = first_fake.get("quat")
        if first_quat is not None:
            real_body.set("quat", first_quat)
        elif "quat" in real_body.attrib:
            del real_body.attrib["quat"]

        # Remove the third hinge joint from real_body
        real_body.remove(joint3)

        # Create ball joint and insert after inertial (if present)
        ball_joint = ET.Element("joint")
        ball_joint.set("name", ball_name)
        ball_joint.set("type", "ball")
        ball_joint.set("damping", joint1.get("damping", "0"))
        inertial_elem = real_body.find("inertial")
        insert_idx = (
            list(real_body).index(inertial_elem) + 1 if inertial_elem is not None else 0
        )
        real_body.insert(insert_idx, ball_joint)

        # Detach real_body from second_fake
        second_fake.remove(real_body)

        # Replace first_fake with real_body in parent
        fake_idx = list(parent_body).index(first_fake)
        parent_body.remove(first_fake)
        parent_body.insert(fake_idx, real_body)

    return mjcf

import xml.etree.ElementTree as ET
from typing import List
import idyntree.swig as idyntree
import numpy as np


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
    mjcf: ET.Element, joint1: str, joint2: str, name: str = None
) -> ET.Element:
    """Add a joint equality constraint between two joints.

    Args:
        mjcf (ET.Element): The mjcf file.
        joint1 (str): The first joint.
        joint2 (str): The second joint.
        name (str): The name of the equality constraint (default: f"{joint1}_{joint2}").
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


def add_sites_for_ft(mjcf: ET.Element, robot_urdf: ET.Element) -> ET.Element:
    for fixed_link in robot_urdf.findall(".//joint[@type='fixed']"):
        if "_ft_fixed" not in fixed_link.attrib["name"]:
            continue

        parent = fixed_link.find("parent").attrib["link"]
        body = mjcf.find(f".//body[@name='{parent}']")
        if body is None:
            print(f"Body {parent} not found in mjcf")
            continue

        # Create <site> element
        site = ET.SubElement(body, "site")
        site.set(
            "name",
            fixed_link.attrib["name"],
        )
        site.set("pos", fixed_link.find("origin").attrib["xyz"])
        rpy = list(map(float, fixed_link.find("origin").attrib["rpy"].split()))
        rotation = idyntree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
        quaternion = rotation.asQuaternion()
        print(quaternion)

        site.set(
            "quat", f"{quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]}"
        )
    return mjcf


def add_sites_for_imu(mjcf: ET.Element, robot_urdf: ET.Element) -> ET.Element:
    for fixed_link in robot_urdf.findall(".//joint[@type='fixed']"):
        if "_imu" not in fixed_link.attrib["name"]:
            continue

        parent = fixed_link.find("parent").attrib["link"]
        body = mjcf.find(f".//body[@name='{parent}']")
        if body is None:
            print(f"Body {parent} not found in mjcf")
            continue

        # Create <site> element
        site = ET.SubElement(body, "site")
        site.set(
            "name",
            fixed_link.attrib["name"],
        )
        site.set("pos", fixed_link.find("origin").attrib["xyz"])
        rpy = list(map(float, fixed_link.find("origin").attrib["rpy"].split()))
        rotation = idyntree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
        quaternion = rotation.asQuaternion()
        print(quaternion)

        site.set(
            "quat", f"{quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]}"
        )

    for sensor in robot_urdf.findall(".//sensor[@type='accelerometer']"):
        parent = sensor.find("parent").attrib["link"]
        body = mjcf.find(f".//body[@name='{parent}']")
        if body is None:
            print(f"Body {parent} not found in mjcf")
            continue

        # Create <site> element
        site = ET.SubElement(body, "site")
        site.set(
            "name",
            sensor.attrib["name"],
        )
        site.set("pos", sensor.find("origin").attrib["xyz"])
        rpy = list(map(float, sensor.find("origin").attrib["rpy"].split()))
        rotation = idyntree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
        quaternion = rotation.asQuaternion()
        print(quaternion)

        site.set(
            "quat", f"{quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]}"
        )
    return mjcf


def add_sites_to_body(
    mjcf: ET.Element, urdf: ET.Element, parent_body: str, target_link: str
) -> ET.Element:
    """
    Add frames to the specified body in the MJCF file based on the URDF joints.

    Args:
        mjcf (ET.Element): The MJCF file as ElementTree.
        urdf (ET.Element): The URDF file as ElementTree.
        parent_body (str): The name of the parent body in the MJCF.

    Returns:
        ET.Element: The modified MJCF file.
    """

    def transform_position(pos, rot, parent_pos, parent_rot):
        transformed_pos = parent_rot * idyntree.Position(
            pos[0], pos[1], pos[2]
        ) + idyntree.Position(parent_pos[0], parent_pos[1], parent_pos[2])
        return [transformed_pos.getVal(i) for i in range(3)]

    def rpy_to_rotation(rpy):
        return idyntree.Rotation.RPY(rpy[0], rpy[1], rpy[2])

    def rotation_to_quaternion(rot):
        quat = rot.asQuaternion()
        return [quat.getVal(0), quat.getVal(1), quat.getVal(2), quat.getVal(3)]

    # Find all fixed joints
    fixed_joints = urdf.findall(".//joint[@type='fixed']")

    # Build a dictionary of parent-child relationships
    link_transformations = {}
    for joint in fixed_joints:
        origin = joint.find("origin")
        if origin is None:
            continue

        # Get the position and RPY values from the joint's origin
        xyz = list(map(float, origin.attrib["xyz"].split()))
        rpy = list(map(float, origin.attrib["rpy"].split()))
        rot = rpy_to_rotation(rpy)

        # Store the transformation
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        link_transformations[child] = (xyz, rot, parent)

    # Function to compute the cumulative transformation for a link
    def get_cumulative_transform(link):
        pos = [0, 0, 0]
        rot = idyntree.Rotation.Identity()
        while link in link_transformations:
            xyz, r, parent = link_transformations[link]
            pos = transform_position(xyz, r, pos, rot)
            rot = r * rot
            link = parent
        return pos, rot

    # Add frames to the MJCF for links connected to l_foot_rear and having "sole" in their names
    for child_link, (xyz, rot, parent_link) in link_transformations.items():
        if "sole" in child_link and parent_link == target_link:
            pos, final_rot = get_cumulative_transform(child_link)
            quat = rotation_to_quaternion(final_rot)
            quat_str = f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"
            pos_str = f"{pos[0]} {pos[1]} {pos[2]}"

            # Find the parent body in the MJCF
            body = mjcf.find(f".//body[@name='{parent_body}']")
            if body is None:
                print(f"Body {parent_body} not found in MJCF")
                continue

            # Create the new frame (site) in the MJCF
            site = ET.SubElement(body, "site")
            site.set("name", child_link)
            site.set("pos", pos_str)
            site.set("quat", quat_str)

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

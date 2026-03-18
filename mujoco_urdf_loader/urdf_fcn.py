import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import resolve_robotics_uri_py as rru


def get_robot_urdf(package: str) -> ET.Element:
    """
    Get the robot urdf from the package.

    Args:
        package (str): The package.

    Returns:
        ET.Element: The robot urdf root element.
    """
    # Get the robot path
    robot_path = rru.resolve_robotics_uri(package)
    # Load the robot urdf
    robot_urdf = ET.parse(robot_path).getroot()

    return robot_urdf


def get_mesh_path(robot_urdf: ET.Element) -> Path:
    """
    Get the mesh path from the robot urdf.

    Args:
        robot_urdf (ET.Element): The robot urdf.

    Returns:
        Path: The mesh path.
    """
    # find the mesh path
    mesh = robot_urdf.find(".//mesh")
    path = mesh.attrib["filename"]
    mesh_path = rru.resolve_robotics_uri(path).parent

    return mesh_path


def remove_gazebo_elements(robot_urdf: ET.Element) -> ET.Element:
    """
    Remove the gazebo elements from the urdf.

    Args:
        robot_urdf (ET.Element): The robot urdf.

    Returns:
        ET.Element: The robot urdf without the gazebo elements.
    """
    new_robot_urdf = copy.deepcopy(robot_urdf)
    for child in new_robot_urdf.findall(".//gazebo/.."):
        for subchild in child.findall("gazebo"):
            child.remove(subchild)

    return new_robot_urdf


def remove_links_and_joints_by_remove_list(
    robot_urdf: ET.Element, to_remove: List[str]
) -> ET.Element:
    """
    Remove the links and joints from the urdf by the remove list.

    The function removes the links and joints that contain the elements in the to_remove list.

    Args:
        robot_urdf (ET.Element): The robot urdf.
        to_remove (List[str]): The list of elements to remove.

    Returns:
        ET.Element: The robot urdf without the elements in the to_remove list.
    """
    new_robot_urdf = copy.deepcopy(robot_urdf)
    for element in to_remove:
        for link in new_robot_urdf.findall(".//link"):
            if element in link.attrib["name"]:
                new_robot_urdf.remove(link)
        for joint in new_robot_urdf.findall(".//joint"):
            if element in joint.attrib["name"]:
                new_robot_urdf.remove(joint)
    return new_robot_urdf


def remove_links_and_joints_by_keep_list(
    robot_urdf: ET.Element, to_keep: List[str]
) -> ET.Element:
    """
    Remove the links and joints from the urdf by the keep list.

    The function removes the links and joints that do not contain the elements in the to_keep list.

    Args:
        robot_urdf (ET.Element): The robot urdf.
        to_keep (List[str]): The list of elements to keep.

    Returns:
        ET.Element: The robot urdf without the elements not in the to_keep list.
    """
    new_robot_urdf = copy.deepcopy(robot_urdf)
    for link in new_robot_urdf.findall(".//link"):
        if all(element not in link.attrib["name"] for element in to_keep):
            new_robot_urdf.remove(link)
    for joint in new_robot_urdf.findall(".//joint"):
        if all(element not in joint.attrib["name"] for element in to_keep):
            new_robot_urdf.remove(joint)
            continue
        # check if the joint is a fixed joint, and skip it
        if "fixed" in joint.attrib["type"]:
            continue
        # check if the joint parent and child are in the to_keep list, if not set the parent to the root_link
        if any(
            element in joint.find("parent").attrib["link"] for element in to_keep
        ) and any(element in joint.find("child").attrib["link"] for element in to_keep):
            continue
        joint.find("parent").set("link", "root_link")
    return new_robot_urdf


def add_mujoco_element(robot_urdf: ET.Element, mesh_path: Path) -> ET.Element:
    """
    Add the mujoco element to the urdf.

    Args:
        robot_urdf (ET.Element): The robot urdf.
        mesh_path (Path): The mesh path.

    Returns:
        ET.Element: The robot urdf with the mujoco element.
    """
    new_robot_urdf = copy.deepcopy(robot_urdf)
    mujoco_elements = ET.SubElement(new_robot_urdf, "mujoco")
    compiler = ET.SubElement(mujoco_elements, "compiler")
    compiler.set(
        "meshdir",
        str(mesh_path),
    )
    return new_robot_urdf


def get_joint_limits(robot_urdf: ET.Element) -> dict:
    """
    Get the joint limits from the urdf.

    Args:
        robot_urdf (ET.Element): The robot urdf.

    Returns:
        dict: The joint limits.
    """
    joint_limits = {}
    for joint in robot_urdf.findall(".//joint"):
        if "fixed" in joint.attrib["type"]:
            continue
        limits = joint.find("limit")
        if limits is None:
            continue
        joint_limits[joint.attrib["name"]] = {
            "lower": float(limits.attrib["lower"]),
            "upper": float(limits.attrib["upper"]),
        }

    return joint_limits


def get_joint_couplings(robot_urdf: ET.Element) -> dict:
    """
    Get the joint couplings from the urdf.

    Args:
        robot_urdf (ET.Element): The robot urdf.
    Returns:
        dict: The joint couplings.
    """
    joint_couplings = {}
    for joint in robot_urdf.findall(".//joint"):
        mimic = joint.find("mimic")
        if mimic is not None:
            joint_couplings[joint.attrib["name"]] = {
                "joint": mimic.attrib["joint"],
                "multiplier": float(mimic.attrib.get("multiplier", 1.0)),
                "offset": float(mimic.attrib.get("offset", 0.0)),
            }

    return joint_couplings


def _is_zero_mass_link(link: ET.Element, tolerance: float = 1e-6) -> bool:
    """Check if a URDF link has zero or negligible mass."""
    inertial = link.find("inertial")
    if inertial is None:
        return True
    mass_elem = inertial.find("mass")
    if mass_elem is None:
        return True
    return abs(float(mass_elem.get("value", "0"))) < tolerance


def set_min_mass_for_zero_mass_links(
    robot_urdf: ET.Element,
    min_mass: float = 1e-8,
    min_inertia: float = 1e-10,
    zero_mass_tolerance: float = 1e-6,
) -> ET.Element:
    """Set a tiny mass/inertia on zero-mass links so MuJoCo accepts them.

    MuJoCo requires moving bodies to have mass > mjMINVAL.  The zero-mass
    dummy links created for spherical-joint chains need a small value to
    survive compilation.

    Args:
        robot_urdf: Root ``<robot>`` element (modified in-place).
        min_mass: Mass value to assign.
        min_inertia: Diagonal inertia value to assign.
        zero_mass_tolerance: Threshold below which a link is considered zero-mass.

    Returns:
        The same ``robot_urdf`` element (modified in-place).
    """
    for link in robot_urdf.findall("link"):
        if not _is_zero_mass_link(link, zero_mass_tolerance):
            continue
        inertial = link.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(link, "inertial")
        mass_elem = inertial.find("mass")
        if mass_elem is None:
            mass_elem = ET.SubElement(inertial, "mass")
        mass_elem.set("value", str(min_mass))
        inertia_elem = inertial.find("inertia")
        if inertia_elem is None:
            inertia_elem = ET.SubElement(inertial, "inertia")
        val = str(min_inertia)
        for attr in ("ixx", "iyy", "izz"):
            inertia_elem.set(attr, val)
        for attr in ("ixy", "ixz", "iyz"):
            inertia_elem.set(attr, "0")
    return robot_urdf


def find_spherical_revolute_joints_in_urdf(
    robot_urdf: ET.Element, zero_mass_tolerance: float = 1e-6
) -> List[str]:
    """Detect 3-revolute spherical joint chains in a URDF and return joint names.

    Searches for chains of 3 consecutive revolute joints where the 2
    intermediate links have zero mass (the pattern produced by
    ``convert_fixed_to_spherical`` / iDynTree convention)::

        parent_link -[rev_x]-> fake_link1 (0 mass) -[rev_y]-> fake_link2 (0 mass) -[rev_z]-> child_link

    Args:
        robot_urdf: Root ``<robot>`` element of the URDF.
        zero_mass_tolerance: Mass threshold for zero-mass detection.

    Returns:
        Flat list of all revolute joint names that belong to spherical chains
        (groups of 3).
    """
    # Build lookup tables
    links = {link.get("name"): link for link in robot_urdf.findall("link")}
    # Map child_link_name -> joint element
    child_to_joint = {}
    for joint in robot_urdf.findall("joint"):
        child_name = joint.find("child").get("link")
        child_to_joint[child_name] = joint
    # Map parent_link_name -> list of joint elements
    parent_to_joints = {}
    for joint in robot_urdf.findall("joint"):
        parent_name = joint.find("parent").get("link")
        parent_to_joints.setdefault(parent_name, []).append(joint)

    spherical_joint_names = []
    visited_joints = set()

    for joint1 in robot_urdf.findall("joint"):
        j1_name = joint1.get("name")
        if j1_name in visited_joints:
            continue
        if joint1.get("type") != "revolute":
            continue

        # First fake link (child of joint1)
        fake1_name = joint1.find("child").get("link")
        fake1 = links.get(fake1_name)
        if fake1 is None or not _is_zero_mass_link(fake1, zero_mass_tolerance):
            continue
        # fake1 must have no visual/collision
        if fake1.find("visual") is not None or fake1.find("collision") is not None:
            continue

        # Joint2: must be the only revolute joint parented by fake1
        children_of_fake1 = [
            j
            for j in parent_to_joints.get(fake1_name, [])
            if j.get("type") == "revolute"
        ]
        if len(children_of_fake1) != 1:
            continue
        joint2 = children_of_fake1[0]

        # Second fake link (child of joint2)
        fake2_name = joint2.find("child").get("link")
        fake2 = links.get(fake2_name)
        if fake2 is None or not _is_zero_mass_link(fake2, zero_mass_tolerance):
            continue
        if fake2.find("visual") is not None or fake2.find("collision") is not None:
            continue

        # Joint3: must be the only revolute joint parented by fake2
        children_of_fake2 = [
            j
            for j in parent_to_joints.get(fake2_name, [])
            if j.get("type") == "revolute"
        ]
        if len(children_of_fake2) != 1:
            continue
        joint3 = children_of_fake2[0]

        # Record the 3 joint names
        names = [j1_name, joint2.get("name"), joint3.get("name")]
        visited_joints.update(names)
        spherical_joint_names.extend(names)

    return spherical_joint_names

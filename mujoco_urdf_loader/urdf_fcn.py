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

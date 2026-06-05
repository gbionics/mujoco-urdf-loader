import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

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


def detect_spherical_joint_groups(
    controlled_joints: List[str],
    rev_joint_prefix: str = "spherical_rev_",
) -> List[Dict]:
    """Detect groups of 3 revolute joints that represent spherical joints.

    The iDynTree convention encodes a spherical joint as three consecutive
    revolute joints named ``{rev_joint_prefix}{base_name}_x``,
    ``{rev_joint_prefix}{base_name}_y``, ``{rev_joint_prefix}{base_name}_z``.

    Args:
        controlled_joints: List of joint names (from config).
        rev_joint_prefix: Naming prefix used for the revolute triplet
            (default: ``"spherical_rev_"``).

    Returns:
        List of dicts, each with keys:
        - ``base_name`` (str): e.g. ``"r_motor_rod_in"``
        - ``joint_x``, ``joint_y``, ``joint_z`` (str): full joint names
    """
    suffix_re = re.compile(
        rf"^{re.escape(rev_joint_prefix)}(.+)_(x|y|z)$"
    )

    # Collect base_name -> {axis: joint_name}
    candidates: Dict[str, Dict[str, str]] = {}
    for jname in controlled_joints:
        m = suffix_re.match(jname)
        if m:
            base_name, axis = m.group(1), m.group(2)
            candidates.setdefault(base_name, {})[axis] = jname

    groups = []
    for base_name, axes in candidates.items():
        if {"x", "y", "z"} <= set(axes):
            groups.append(
                {
                    "base_name": base_name,
                    "joint_x": axes["x"],
                    "joint_y": axes["y"],
                    "joint_z": axes["z"],
                }
            )
    return groups


def collapse_spherical_revolute_triplets(
    robot_urdf: ET.Element,
    spherical_groups: List[Dict],
    fake_link_prefix: str = "spherical_fake_",
) -> Tuple[ET.Element, Dict[str, str]]:
    """Collapse 3-revolute spherical joint triplets into single placeholder joints.

    For each detected spherical group, this function:
    1. Removes the two zero-mass dummy links (``{fake_link_prefix}{base_name}_link1``
       and ``_link2``).
    2. Removes the ``_y`` and ``_z`` revolute joints.
    3. Rewrites the ``_x`` joint so its ``<child>`` points directly to the
       final child link (the child of the ``_z`` joint) and changes its type
       to ``"continuous"`` (unlimited hinge).

    The caller should later convert the resulting hinge joint into a MuJoCo
    ``ball`` joint in the MJCF post-processing step.

    Args:
        robot_urdf: URDF root element (modified **in-place** and also returned).
        spherical_groups: Output of :func:`detect_spherical_joint_groups`.
        fake_link_prefix: Prefix used for dummy link names.

    Returns:
        Tuple of (modified URDF element, mapping from placeholder joint name
        to ``base_name``).
    """
    urdf = copy.deepcopy(robot_urdf)
    ball_joint_map: Dict[str, str] = {}

    # Index joints and links for fast lookup
    joints_by_name = {j.attrib["name"]: j for j in urdf.findall(".//joint")}
    links_by_name = {l.attrib["name"]: l for l in urdf.findall(".//link")}

    for group in spherical_groups:
        base_name = group["base_name"]
        jx_name = group["joint_x"]
        jy_name = group["joint_y"]
        jz_name = group["joint_z"]

        jx = joints_by_name.get(jx_name)
        jy = joints_by_name.get(jy_name)
        jz = joints_by_name.get(jz_name)

        if jx is None or jy is None or jz is None:
            raise ValueError(
                f"Spherical group '{base_name}': could not find all three "
                f"revolute joints ({jx_name}, {jy_name}, {jz_name}) in the URDF."
            )

        # The final child is the child of the _z joint (e.g. r_rod_in)
        final_child = jz.find("child").attrib["link"]

        # Dummy link names
        dummy_link1 = f"{fake_link_prefix}{base_name}_link1"
        dummy_link2 = f"{fake_link_prefix}{base_name}_link2"

        # --- Remove dummy links ---
        for dname in (dummy_link1, dummy_link2):
            link_elem = links_by_name.get(dname)
            if link_elem is not None:
                urdf.remove(link_elem)

        # --- Remove _y and _z joints ---
        for jname in (jy_name, jz_name):
            joint_elem = joints_by_name.get(jname)
            if joint_elem is not None:
                urdf.remove(joint_elem)

        # --- Rewrite _x joint as placeholder continuous joint ---
        jx.attrib["type"] = "continuous"
        jx.find("child").set("link", final_child)
        # Remove axis element (ball joints are axis-free)
        axis_elem = jx.find("axis")
        if axis_elem is not None:
            jx.remove(axis_elem)
        # Remove dynamics if present (not needed for placeholder)
        dynamics_elem = jx.find("dynamics")
        if dynamics_elem is not None:
            jx.remove(dynamics_elem)

        ball_joint_map[jx_name] = base_name

    return urdf, ball_joint_map

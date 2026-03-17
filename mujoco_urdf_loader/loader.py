import dataclasses
import math
import tempfile
import xml.etree.ElementTree as ET
from enum import Enum
from typing import List, Union

import idyntree.bindings as idyn

from mujoco_urdf_loader.generator import load_urdf_into_mjcf
from mujoco_urdf_loader.mjcf_fcn import (
    add_position_actuator,
    add_torque_actuator,
    separate_left_right_collision_groups,
)
from mujoco_urdf_loader.urdf_fcn import (
    add_mujoco_element,
    get_mesh_path,
    remove_gazebo_elements,
)


class ControlMode(Enum):
    POSITION = "position"
    TORQUE = "torque"
    VELOCITY = "velocity"


@dataclasses.dataclass
class URDFtoMuJoCoLoaderCfg:
    controlled_joints: List[str]
    control_modes: Union[None, List[ControlMode]] = None
    stiffness: Union[None, List[float]] = None
    damping: Union[None, List[float]] = None
    all_missing_joints_as_sites: bool = False


class URDFtoMuJoCoLoader:
    def __init__(self, mjcf: str, cfg: URDFtoMuJoCoLoaderCfg):
        """
        Initialize the URDF to Mujoco converter.

        Args:
            mjcf (str): The MuJoCo string.
            joints (List[str]): The list of joints to command.
        """
        self.mjcf = mjcf
        self.controlled_joints = cfg.controlled_joints
        if cfg.control_modes is None:
            self.control_mode = {joint: ControlMode.TORQUE for joint in cfg.controlled_joints}
        else:
            self.control_mode = {joint: mode for joint, mode in zip(cfg.controlled_joints, cfg.control_modes)}
        self.set_controlled_joints(cfg.controlled_joints)

    @staticmethod
    def load_urdf(urdf_path: str, mesh_path: str, cfg: URDFtoMuJoCoLoaderCfg):
        """
        Load the URDF from the file.

        Args:
            urdf_path (Path): The URDF file path.
            cfg (URDFtoMuJoCoLoaderCfg): The configuration containing the controlled joints, control modes, stiffness and damping.

        Returns:
            str: The URDF string.
        """
        original_urdf = ET.parse(urdf_path).getroot()
        original_urdf = remove_gazebo_elements(original_urdf)

        urdf_string = URDFtoMuJoCoLoader.simplify_urdf(urdf_path, cfg.controlled_joints, cfg.stiffness, cfg.damping)
        urdf_string = remove_gazebo_elements(urdf_string)
        urdf_string = add_mujoco_element(urdf_string, mesh_path)
        mjcf = load_urdf_into_mjcf(urdf_string)
        mjcf = separate_left_right_collision_groups(mjcf)

        missing_joint_sites = URDFtoMuJoCoLoader.get_missing_joint_sites(
            original_urdf,
            mjcf,
            all_missing_joints_as_sites=cfg.all_missing_joints_as_sites,
        )

        loader = URDFtoMuJoCoLoader(mjcf, cfg)
        loader.add_sites_for_missing_joints(missing_joint_sites)
        return loader

    @staticmethod
    def get_missing_joint_sites(
        robot_urdf: ET.Element,
        mjcf: ET.Element,
        all_missing_joints_as_sites: bool = False,
    ) -> List[dict]:
        """Extract metadata for URDF joints that are missing in MJCF.

        Args:
            robot_urdf (ET.Element): The URDF root element.
            mjcf (ET.Element): The MJCF root element.
            all_missing_joints_as_sites (bool): If True, create sites for all URDF joints
                whose names are missing in MJCF.

        Returns:
            List[dict]: Site descriptors with joint name, parent/child links and origin.
        """
        if not all_missing_joints_as_sites:
            return []

        all_joint_elements = robot_urdf.findall(".//joint")
        urdf_joint_elements = {
            joint.attrib["name"]: joint
            for joint in all_joint_elements
            if "name" in joint.attrib
        }
        mjcf_joint_names = {
            joint.attrib["name"]
            for joint in mjcf.findall(".//joint")
            if "name" in joint.attrib
        }

        missing_joint_names = [
            joint_name
            for joint_name in urdf_joint_elements
            if joint_name not in mjcf_joint_names
        ]

        joint_by_child_link = {}
        for joint in all_joint_elements:
            child = joint.find("child")
            if child is not None and "link" in child.attrib:
                joint_by_child_link[child.attrib["link"]] = joint

        fixed_joint_sites = []
        for fixed_joint_name in missing_joint_names:
            joint = urdf_joint_elements.get(fixed_joint_name)
            if joint is None:
                raise ValueError(
                    f"Joint {fixed_joint_name} not found in the URDF model."
                )

            parent_link = joint.find("parent").attrib["link"]
            child_link = joint.find("child").attrib["link"]
            origin = joint.find("origin")
            xyz = origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            rpy = origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0"

            # Build all ancestor-link candidates with transform from ancestor link frame
            # to the fixed-joint site frame. This is needed when MuJoCo lumps nested
            # fixed joints and intermediate bodies disappear.
            ancestor_candidates = []
            current_link = child_link
            current_rot_to_site = URDFtoMuJoCoLoader.identity_rot()
            current_pos_to_site = [0.0, 0.0, 0.0]

            ancestor_candidates.append(
                {
                    "link": current_link,
                    "xyz": URDFtoMuJoCoLoader.vec_to_str(current_pos_to_site),
                    "quat": URDFtoMuJoCoLoader.rot_to_quat_str(current_rot_to_site),
                }
            )

            while current_link in joint_by_child_link:
                parent_joint = joint_by_child_link[current_link]
                parent_link_candidate = parent_joint.find("parent").attrib["link"]
                parent_origin = parent_joint.find("origin")
                parent_xyz = (
                    parent_origin.attrib.get("xyz", "0 0 0")
                    if parent_origin is not None
                    else "0 0 0"
                )
                parent_rpy = (
                    parent_origin.attrib.get("rpy", "0 0 0")
                    if parent_origin is not None
                    else "0 0 0"
                )

                rot_parent_current = URDFtoMuJoCoLoader.urdf_rpy_to_rot(parent_rpy)
                pos_parent_current = URDFtoMuJoCoLoader.str_to_vec(parent_xyz)

                current_pos_to_site = URDFtoMuJoCoLoader.compose_pos(
                    rot_parent_current,
                    pos_parent_current,
                    current_pos_to_site,
                )
                current_rot_to_site = URDFtoMuJoCoLoader.compose_rot(
                    rot_parent_current,
                    current_rot_to_site,
                )

                ancestor_candidates.append(
                    {
                        "link": parent_link_candidate,
                        "xyz": URDFtoMuJoCoLoader.vec_to_str(current_pos_to_site),
                        "quat": URDFtoMuJoCoLoader.rot_to_quat_str(current_rot_to_site),
                    }
                )

                current_link = parent_link_candidate

            fixed_joint_sites.append(
                {
                    "name": fixed_joint_name,
                    "parent_link": parent_link,
                    "child_link": child_link,
                    "xyz": xyz,
                    "rpy": rpy,
                    "ancestor_candidates": ancestor_candidates,
                }
            )

        return fixed_joint_sites

    @staticmethod
    def urdf_rpy_to_quat(rpy: str) -> str:
        """Convert URDF RPY (extrinsic XYZ) to MuJoCo quaternion (w x y z)."""
        return URDFtoMuJoCoLoader.rot_to_quat_str(URDFtoMuJoCoLoader.urdf_rpy_to_rot(rpy))

    @staticmethod
    def identity_rot() -> List[List[float]]:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    @staticmethod
    def str_to_vec(xyz: str) -> List[float]:
        return list(map(float, xyz.split()))

    @staticmethod
    def vec_to_str(vec: List[float]) -> str:
        return f"{vec[0]} {vec[1]} {vec[2]}"

    @staticmethod
    def matmul3x3(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        return [
            [
                a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
                a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
            ],
            [
                a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
                a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
            ],
            [
                a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
                a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
                a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
            ],
        ]

    @staticmethod
    def matvec3(m: List[List[float]], v: List[float]) -> List[float]:
        return [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]

    @staticmethod
    def compose_rot(r_ab: List[List[float]], r_bc: List[List[float]]) -> List[List[float]]:
        return URDFtoMuJoCoLoader.matmul3x3(r_ab, r_bc)

    @staticmethod
    def compose_pos(
        r_ab: List[List[float]],
        p_ab: List[float],
        p_bc: List[float],
    ) -> List[float]:
        r_ab_p_bc = URDFtoMuJoCoLoader.matvec3(r_ab, p_bc)
        return [
            p_ab[0] + r_ab_p_bc[0],
            p_ab[1] + r_ab_p_bc[1],
            p_ab[2] + r_ab_p_bc[2],
        ]

    @staticmethod
    def urdf_rpy_to_rot(rpy: str) -> List[List[float]]:
        """URDF rpy uses fixed/extrinsic XYZ; matrix is Rz(yaw) * Ry(pitch) * Rx(roll)."""
        roll, pitch, yaw = map(float, rpy.split())

        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        rx = [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]
        ry = [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
        rz = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]

        return URDFtoMuJoCoLoader.matmul3x3(rz, URDFtoMuJoCoLoader.matmul3x3(ry, rx))

    @staticmethod
    def rot_to_quat_str(rot: List[List[float]]) -> str:
        qw, qx, qy, qz = URDFtoMuJoCoLoader.rot_to_quat(rot)
        return f"{qw} {qx} {qy} {qz}"

    @staticmethod
    def rot_to_quat(rot: List[List[float]]) -> List[float]:
        trace = rot[0][0] + rot[1][1] + rot[2][2]
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (rot[2][1] - rot[1][2]) / s
            qy = (rot[0][2] - rot[2][0]) / s
            qz = (rot[1][0] - rot[0][1]) / s
        elif rot[0][0] > rot[1][1] and rot[0][0] > rot[2][2]:
            s = math.sqrt(1.0 + rot[0][0] - rot[1][1] - rot[2][2]) * 2.0
            qw = (rot[2][1] - rot[1][2]) / s
            qx = 0.25 * s
            qy = (rot[0][1] + rot[1][0]) / s
            qz = (rot[0][2] + rot[2][0]) / s
        elif rot[1][1] > rot[2][2]:
            s = math.sqrt(1.0 + rot[1][1] - rot[0][0] - rot[2][2]) * 2.0
            qw = (rot[0][2] - rot[2][0]) / s
            qx = (rot[0][1] + rot[1][0]) / s
            qy = 0.25 * s
            qz = (rot[1][2] + rot[2][1]) / s
        else:
            s = math.sqrt(1.0 + rot[2][2] - rot[0][0] - rot[1][1]) * 2.0
            qw = (rot[1][0] - rot[0][1]) / s
            qx = (rot[0][2] + rot[2][0]) / s
            qy = (rot[1][2] + rot[2][1]) / s
            qz = 0.25 * s

        norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if norm == 0.0:
            return [1.0, 0.0, 0.0, 0.0]
        return [qw / norm, qx / norm, qy / norm, qz / norm]

    @staticmethod
    def simplify_urdf(
        urdf_path: str,
        joints: List[str],
        stiffness: List[float] = None,
        damping: List[float] = None,
    ):
        """
        Simplify the URDF using iDynTree.

        Args:
            urdf_path (str): The URDF string.
            joints (List[str]): The list of joints to command.
            stiffness (List[float]): The list of stiffness values.
            damping (List[float]): The list of damping values.

        Returns:
            str: The simplified URDF string.
        """

        # Load the URDF model
        model_loader = idyn.ModelLoader()
        if not model_loader.loadReducedModelFromFile(urdf_path, joints):
            raise ValueError(f"Error loading the URDF model from {urdf_path}. Check the file path or if the joints are correct.")
        model = model_loader.model()

        if stiffness is not None:
            for i in range(model.getNrOfJoints()):
                joint = model.getJoint(i)
                for dof in range(joint.getNrOfDOFs()):
                    joint.setStaticFriction(dof, stiffness[i])

        if damping is not None:
            for i in range(model.getNrOfJoints()):
                joint = model.getJoint(i)
                for dof in range(joint.getNrOfDOFs()):
                    joint.setDamping(dof, damping[i])


        # Save the simplified model
        model_saver = idyn.ModelExporter()
        model_saver.init(model)

        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name
            model_saver.exportModelToFile(temp_path)

        tree = ET.parse(temp_path)
        root = tree.getroot()

        URDFtoMuJoCoLoader.connect_root_to_world(root)

        return root

    @staticmethod
    def connect_root_to_world(root):
        """
        Connect the root link to the world.

        Args:
            root (ET.Element): The root element.
        """

        # Find all the links and joints in the URDF
        links = {link.attrib["name"]: link for link in root.findall(".//link")}
        joints = root.findall(".//joint")
        # Find child and parent links for each joint
        child_links = {joint.find("child").attrib["link"] for joint in joints}
        parent_links = {joint.find("parent").attrib["link"] for joint in joints}
        # The root link is a parent link that is not a child link
        root_link = next(link for link in links if link not in child_links)
        # Find the joint that has the root link as parent
        fixed_joint_found = False
        for joint in joints:
            parent_link = joint.find("parent").attrib["link"]
            if parent_link == root_link:
                # Check if the joint is fixed
                if joint.attrib["type"] == "fixed":
                    # Change the joint type to floating
                    joint.attrib["type"] = "floating"
                    print(f"Modified joint {joint.attrib['name']} to type floating.")
                    fixed_joint_found = True
                break

        if not fixed_joint_found:
            raise ValueError("No fixed joint found that can be modified to floating.")

    def set_control_mode(self, joint: Union[str, List[str]], mode: ControlMode):
        """
        Set the control mode for the joint.

        Args:
            joint (str): The joint name.
            mode (ControlMode): The control mode.
        """
        if isinstance(joint, str):
            self.control_mode[joint] = mode
        elif isinstance(joint, list):
            for j in joint:
                self.control_mode[j] = mode
        else:
            raise ValueError("joint must be a string or a list of strings.")

    def add_actuator(self, joint: str, control_mode: ControlMode):
        """
        Add an actuator to the MJCF model.

        Args:
            joint (str): The joint name.
            control_mode (ControlMode): The control mode.
        """
        if control_mode == ControlMode.POSITION:
            joint_mjcf = self.mjcf.find(f".//joint[@name='{joint}']")
            ctrlrange = list(map(float, joint_mjcf.attrib["range"].split()))
            add_position_actuator(self.mjcf, joint=joint, ctrlrange=ctrlrange)
        elif control_mode == ControlMode.TORQUE:
            add_torque_actuator(self.mjcf, joint=joint, ctrlrange=None)
        elif control_mode == ControlMode.VELOCITY:
            raise NotImplementedError("Velocity control is not implemented yet.")
        else:
            raise ValueError("Control mode not recognized.")

    def set_controlled_joints(self, joints: List[str]):
        """
        Set the controlled joints.

        Args:
            joints (List[str]): The list of joints.
        """
        self.controlled_joints = joints
        joint_elements = {joint.attrib["name"]: joint for joint in self.mjcf.findall(".//joint")}

        for controlled_joint in self.controlled_joints:
            joint_element = joint_elements.get(controlled_joint)
            if joint_element is not None:
                self.add_actuator(controlled_joint, self.control_mode[controlled_joint])
            else:
                raise ValueError(
                    f"Joint {controlled_joint} not found in the MJCF model."
                )

    def add_sites_for_missing_joints(self, fixed_joint_sites: List[dict]):
        """Add one MJCF site for each configured missing URDF joint.

        The site is attached to the fixed joint child link body when available.
        If the child body is not present in MJCF, the site is attached to the
        parent link body at the URDF joint origin.
        """
        for fixed_joint_site in fixed_joint_sites:
            site_name = fixed_joint_site["name"]

            # Skip if site already exists
            if self.mjcf.find(f".//site[@name='{site_name}']") is not None:
                continue

            candidates = fixed_joint_site.get("ancestor_candidates")
            if not candidates:
                candidates = [
                    {
                        "link": fixed_joint_site["child_link"],
                        "xyz": "0 0 0",
                        "quat": "1 0 0 0",
                    },
                    {
                        "link": fixed_joint_site["parent_link"],
                        "xyz": fixed_joint_site["xyz"],
                        "quat": self.urdf_rpy_to_quat(fixed_joint_site["rpy"]),
                    },
                ]

            attached = False
            for candidate in candidates:
                body = self.mjcf.find(f".//body[@name='{candidate['link']}']")
                if body is None:
                    continue

                ET.SubElement(
                    body,
                    "site",
                    {
                        "name": site_name,
                        "pos": candidate["xyz"],
                        "quat": candidate["quat"],
                    },
                )
                attached = True
                break

            if not attached:
                raise ValueError(
                    f"Cannot create site for fixed joint {site_name}: "
                    "no surviving ancestor body found in MJCF."
                )

    def get_mjcf(self):
        """
        Get the Mujoco XML string.

        Returns:
            str: The Mujoco XML string.
        """
        return self.mjcf

    def get_mjcf_string(self, pretty: bool = False):
        """
        Get the Mujoco XML string.

        Args:
            pretty (bool): If True, return an indented, human-readable XML string.

        Returns:
            str: The Mujoco XML string.
        """
        mjcf = self.mjcf
        if pretty:
            mjcf = ET.fromstring(ET.tostring(self.mjcf, encoding="unicode", method="xml"))
            ET.indent(mjcf, space="  ")
        return ET.tostring(mjcf, encoding="unicode", method="xml")

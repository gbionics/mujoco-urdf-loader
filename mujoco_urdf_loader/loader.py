import idyntree.bindings as idyn
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List
import resolve_robotics_uri_py as rru
import tempfile
import dataclasses
import mujoco
import numpy as np

from mujoco_urdf_loader.urdf_fcn import (
    add_mujoco_element,
    get_mesh_path,
    remove_gazebo_elements,
)
from mujoco_urdf_loader.mjcf_fcn import (
    add_position_actuator,
    add_torque_actuator,
    separate_left_right_collision_groups,
)
from mujoco_urdf_loader.generator import load_urdf_into_mjcf

from enum import Enum

class ControlMode(Enum):
    POSITION = "position"
    TORQUE = "torque"
    VELOCITY = "velocity"


class URDFtoMuJoCoLoader:
    def __init__(self, mjcf: str, joints: List[str]):
        """
        Initialize the URDF to Mujoco converter.

        Args:
            mjcf (str): The MuJoCo string.
            joints (List[str]): The list of joints to command.
        """
        self.mjcf = mjcf
        self.controlled_joints = joints
        self.control_mode = {joint: ControlMode.TORQUE for joint in joints}

    @staticmethod
    def load_urdf(urdf_path: Path, joints: List[str], stiffness: List[float] = None, damping: List[float] = None):
        """
        Load the URDF from the file.

        Args:
            urdf_path (Path): The URDF file path.
            joints (List[str]): The list of joints to command.
            stiffness (List[float]): The list of stiffness values.
            damping (List[float]): The list of damping values.

        Returns:
            str: The URDF string.
        """
        urdf_string = URDFtoMuJoCoLoader.simplify_urdf(urdf_path, joints, stiffness, damping)
        mesh_path = get_mesh_path(urdf_string)
        urdf_string = remove_gazebo_elements(urdf_string)
        urdf_string = add_mujoco_element(urdf_string, mesh_path)
        mjcf = load_urdf_into_mjcf(urdf_string)
        mjcf = separate_left_right_collision_groups(mjcf)
        return URDFtoMuJoCoLoader(mjcf, joints)

    @staticmethod
    def simplify_urdf(urdf_path: str, joints: List[str], stiffness: List[float] = None, damping: List[float] = None):
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
        model_loader.loadReducedModelFromFile(urdf_path, joints)
        model = model_loader.model()

        if stiffness is not None:
            for i in range(model.getNrOfJoints()):
                joint = model.getJoint(i)
                for dof in range(joint.getNrOfDOFs()):
                    joint.setStiffness(dof, stiffness[i])

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

        # Find the joint with name "base_link_fixed_joint" and change its type to "floating"
        for joint in root.findall(".//joint"):
            if joint.attrib.get("name") == "base_link_fixed_joint":
                joint.attrib["type"] = "floating"
                break

        return root
    
    def set_control_mode(self, joint: str, mode: ControlMode):
        """
        Set the control mode for the joint.

        Args:
            joint (str): The joint name.
            mode (ControlMode): The control mode.
        """
        self.control_mode[joint] = mode

    def set_controlled_joints(self, joints: List[str]):
        """
        Set the controlled joints.

        Args:
            joints (List[str]): The list of joints.
        """
        self.controlled_joints = joints
        for controlled_joint in self.controlled_joints:
            for joint in self.mjcf.findall(".//joint"):
                if joint.attrib.get("name") == controlled_joint:
                    if self.control_mode[controlled_joint] == ControlMode.POSITION:
                        ctrlrange = joint.attrib["range"]
                        add_position_actuator(self.mjcf, 
                                                joint=joint.attrib["name"],
                                                ctrlrange=[float(ctrlrange.split()[0]), float(ctrlrange.split()[1])])
                    elif self.control_mode[controlled_joint] == ControlMode.TORQUE:
                        ctrlrange = joint.attrib["range"]
                        add_torque_actuator(self.mjcf,
                                            joint=joint.attrib["name"],
                                            ctrlrange=[
                                                -300.0,
                                                300.0,
                                            ])
                    elif self.control_mode[controlled_joint] == ControlMode.VELOCITY:
                        raise NotImplementedError("Velocity control is not implemented yet.")
                    else:
                        raise ValueError("Invalid control mode.")
                    
    def get_mjcf(self):
        """
        Get the Mujoco XML string.

        Returns:
            str: The Mujoco XML string.
        """
        return self.mjcf
    
    def get_mjcf_string(self):
        """
        Get the Mujoco XML string.

        Returns:
            str: The Mujoco XML string.
        """
        return ET.tostring(self.mjcf, encoding="unicode", method="xml")

import dataclasses
import logging
import numpy as np
import mujoco

@dataclasses.dataclass
class MujocoWrapper:
    mjcf: str
    model: mujoco.MjModel = None
    data: mujoco.MjData = None

    def __post_init__(self):
        self.model = mujoco.MjModel.from_xml_string(self.mjcf)
        self.data = mujoco.MjData(self.model)
        self.joint_names = [self.model.joint(j).name for j in range(self.model.njnt) if self.model.joint(j).name != "base_link_fixed_joint"]
        self.actuator_names = [self.model.actuator(a).name for a in range(self.model.nu)]
        logging.info(f"Actuator loaded: {self.actuator_names}")
        self.initialize_joint_mapping()

    def initialize_joint_mapping(self):
        self.joint_to_actuator_mapping = np.zeros((self.model.nu, self.model.nq - 7))
        self.actuator_to_joint_mapping = np.zeros((self.model.nq - 7, self.model.nu))
        for i, joint_name in enumerate(self.joint_names):
            for j, actuator_name in enumerate(self.actuator_names):
                if joint_name in actuator_name:
                    self.joint_to_actuator_mapping[j, i] = 1
                    self.actuator_to_joint_mapping[i, j] = 1
        logging.info("Joint to actuator mapping initialized")

    def set_control(self, control: np.ndarray):
        """
        Set the control input.

        Args:
            control (np.ndarray): The control input.
        """
        if control.shape != self.data.ctrl.shape:
            raise ValueError(f"Control input shape {control.shape} does not match the expected shape {self.data.ctrl.shape}.")
        self.data.ctrl[:] = control
    
    @property
    def joint_positions(self):
        """
        Get the joint positions.

        Returns:
            np.ndarray: The joint positions.
        """
        return self.joint_to_actuator_mapping @ self.data.qpos[7:]
    
    @property
    def joint_velocities(self):
        """
        Get the joint velocities.

        Returns:
            np.ndarray: The joint velocities.
        """
        return self.joint_to_actuator_mapping @ self.data.qvel[6:]
    
    @property
    def base_position(self):
        """
        Get the base position.

        Returns:
            np.ndarray: The base position.
        """
        return self.data.qpos[:3]
    
    @property
    def base_orientation(self):
        """
        Get the base orientation.

        Returns:
            np.ndarray: The base orientation in quaternion (scalar first).
        """
        return self.data.qpos[3:7]

    @property
    def base_velocity(self):
        """
        Get the base velocity, linear and angular. MuJoCo returns the base velocity in the mixed representation.

        Returns:
            np.ndarray: The base velocity (linear and angular).
        """
        return self.data.qvel[:6]
    
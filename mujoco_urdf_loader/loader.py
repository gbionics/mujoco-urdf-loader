import dataclasses
import tempfile
import xml.etree.ElementTree as ET
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import idyntree.bindings as idyn
import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_urdf_loader.generator import load_urdf_into_mjcf
from mujoco_urdf_loader.mjcf_fcn import (
    add_position_actuator,
    add_torque_actuator,
    separate_left_right_collision_groups,
    add_framequat_sensor,
    add_gyro_sensor,
    add_camera_to_site,
    add_equality_constraints_for_sites,
    convert_hinge_to_ball_joints,
)
from mujoco_urdf_loader.urdf_fcn import (
    add_mujoco_element,
    remove_gazebo_elements,
    detect_spherical_joint_groups,
    collapse_spherical_revolute_triplets,
)


class ControlMode(Enum):
    POSITION = "position"
    TORQUE = "torque"
    VELOCITY = "velocity"


@dataclasses.dataclass
class FrameQuatSensorCfg:
    objname: str
    objtype: str = "site"
    name: str = None


@dataclasses.dataclass
class GyroSensorCfg:
    site: str
    name: str = None


@dataclasses.dataclass
class CameraCfg:
    site: str
    fovy: float
    name: str


@dataclasses.dataclass
class EqualityConstraintCfg:
    """Configuration for a connect/weld equality constraint between two sites."""

    site1: str
    site2: str
    constraint_type: str = "connect"
    solimp: Optional[List[float]] = None
    solref: Optional[List[float]] = None


@dataclasses.dataclass
class URDFtoMuJoCoLoaderCfg:
    observed_joints: Union[None, List[str]] = None
    actuated_joints: Union[None, List[str]] = None
    control_modes: Union[None, List[ControlMode]] = None
    stiffness: Union[None, List[float]] = None
    damping: Union[None, List[float]] = None
    joint_damping: Union[None, List[float]] = None
    joint_frictionloss: Union[None, List[float]] = None
    armature: Union[None, List[float]] = None
    all_missing_joints_as_sites: bool = False
    framequat_sensors_cfg: Union[
        None, List[Union[FrameQuatSensorCfg, Dict[str, Any]]]
    ] = None
    gyro_sensors_cfg: Union[None, List[Union[GyroSensorCfg, Dict[str, Any]]]] = None
    cameras_cfg: Union[None, List[Union[CameraCfg, Dict[str, Any]]]] = None
    equality_constraints_cfg: Union[
        None, List[Union[EqualityConstraintCfg, Dict[str, Any]]]
    ] = None
    ball_joint_damping: float = 0.0
    ball_joint_armature: float = 0.0
    ball_joint_frictionloss: float = 0.0


class URDFtoMuJoCoLoader:
    def __init__(self, mjcf: str, cfg: URDFtoMuJoCoLoaderCfg):
        """
        Initialize the URDF to Mujoco converter.

        Args:
            mjcf (str): The MuJoCo string.
            cfg (URDFtoMuJoCoLoaderCfg): Loader configuration.
        """
        normalized_cfg = self._normalize_cfg(cfg)

        self.mjcf = mjcf
        self.observed_joints = normalized_cfg.observed_joints
        self.actuated_joints = normalized_cfg.actuated_joints

        self.control_mode = {
            joint: mode
            for joint, mode in zip(self.actuated_joints, normalized_cfg.control_modes)
        }

        self.set_observed_joint_dynamics(
            damping=normalized_cfg.joint_damping,
            frictionloss=normalized_cfg.joint_frictionloss,
        )

        self.set_actuated_joints(
            normalized_cfg.actuated_joints,
            stiffness=normalized_cfg.stiffness,
            damping=normalized_cfg.damping,
        )

    @staticmethod
    def _normalize_cfg(cfg: URDFtoMuJoCoLoaderCfg) -> URDFtoMuJoCoLoaderCfg:
        """Normalize and validate joint/control related arrays."""
        observed_joints = cfg.observed_joints

        if observed_joints is None:
            raise ValueError("observed_joints must be provided.")
        observed_joints = list(observed_joints)

        actuated_joints = (
            list(cfg.actuated_joints)
            if cfg.actuated_joints is not None
            else list(observed_joints)
        )

        observed_set = set(observed_joints)
        invalid_actuated = [j for j in actuated_joints if j not in observed_set]
        if invalid_actuated:
            raise ValueError(
                "actuated_joints must be a subset of observed_joints. "
                f"Invalid entries: {invalid_actuated}"
            )

        if cfg.control_modes is None:
            control_modes = [ControlMode.TORQUE] * len(actuated_joints)
        else:
            if len(cfg.control_modes) != len(actuated_joints):
                raise ValueError(
                    f"Length of control_modes ({len(cfg.control_modes)}) must match "
                    f"the number of actuated_joints ({len(actuated_joints)})."
                )
            control_modes = []
            for mode in cfg.control_modes:
                if isinstance(mode, ControlMode):
                    control_modes.append(mode)
                elif isinstance(mode, str):
                    mode_upper = mode.upper()
                    if mode_upper in ControlMode.__members__:
                        control_modes.append(ControlMode[mode_upper])
                    else:
                        try:
                            control_modes.append(ControlMode(mode.lower()))
                        except ValueError as exc:
                            raise ValueError(
                                f"Unsupported control mode: {mode}"
                            ) from exc
                else:
                    raise TypeError(
                        "Each control mode must be a ControlMode enum or string."
                    )

        stiffness = list(cfg.stiffness) if cfg.stiffness is not None else None
        damping = list(cfg.damping) if cfg.damping is not None else None

        has_position = any(mode == ControlMode.POSITION for mode in control_modes)
        if has_position:
            if stiffness is None:
                raise ValueError(
                    "stiffness is required when any actuator uses POSITION control mode."
                )
            if damping is None:
                raise ValueError(
                    "damping is required when any actuator uses POSITION control mode."
                )

        if stiffness is not None and len(stiffness) != len(actuated_joints):
            raise ValueError(
                f"Length of stiffness ({len(stiffness)}) must match "
                f"the number of actuated_joints ({len(actuated_joints)})."
            )

        if damping is not None and len(damping) != len(actuated_joints):
            raise ValueError(
                f"Length of damping ({len(damping)}) must match "
                f"the number of actuated_joints ({len(actuated_joints)})."
            )

        joint_damping = (
            list(cfg.joint_damping) if cfg.joint_damping is not None else None
        )
        if joint_damping is not None and len(joint_damping) != len(observed_joints):
            raise ValueError(
                f"Length of joint_damping ({len(joint_damping)}) must match "
                f"the number of observed_joints ({len(observed_joints)})."
            )

        joint_frictionloss = (
            list(cfg.joint_frictionloss) if cfg.joint_frictionloss is not None else None
        )
        if joint_frictionloss is not None and len(joint_frictionloss) != len(
            observed_joints
        ):
            raise ValueError(
                f"Length of joint_frictionloss ({len(joint_frictionloss)}) must match "
                f"the number of observed_joints ({len(observed_joints)})."
            )

        armature = list(cfg.armature) if cfg.armature is not None else None
        normalized_armature = None
        if armature is not None:
            if len(armature) == len(actuated_joints):
                # Canonical: one armature value per actuated joint.
                normalized_armature = list(armature)
            elif len(armature) == len(observed_joints):
                # Also accept one value per observed joint and project by name to
                # the actuated subset.
                observed_armature = {
                    joint_name: value
                    for joint_name, value in zip(observed_joints, armature)
                }
                normalized_armature = [
                    observed_armature[joint_name] for joint_name in actuated_joints
                ]
            else:
                raise ValueError(
                    f"Length of armature ({len(armature)}) must match "
                    f"the number of actuated_joints ({len(actuated_joints)}) "
                    f"or observed_joints ({len(observed_joints)})."
                )

        return URDFtoMuJoCoLoaderCfg(
            observed_joints=list(observed_joints),
            actuated_joints=list(actuated_joints),
            control_modes=list(control_modes),
            stiffness=stiffness,
            damping=damping,
            joint_damping=joint_damping,
            joint_frictionloss=joint_frictionloss,
            armature=normalized_armature,
            all_missing_joints_as_sites=cfg.all_missing_joints_as_sites,
            framequat_sensors_cfg=cfg.framequat_sensors_cfg,
            gyro_sensors_cfg=cfg.gyro_sensors_cfg,
            cameras_cfg=cfg.cameras_cfg,
            equality_constraints_cfg=cfg.equality_constraints_cfg,
            ball_joint_damping=cfg.ball_joint_damping,
            ball_joint_armature=cfg.ball_joint_armature,
            ball_joint_frictionloss=cfg.ball_joint_frictionloss,
        )

    @staticmethod
    def _filter_cfg_for_removed_joints(
        cfg: URDFtoMuJoCoLoaderCfg, removed_joints: set
    ) -> URDFtoMuJoCoLoaderCfg:
        if not removed_joints:
            return cfg

        filtered_observed_indices = [
            i
            for i, joint in enumerate(cfg.observed_joints)
            if joint not in removed_joints
        ]
        filtered_observed = [cfg.observed_joints[i] for i in filtered_observed_indices]

        filtered_actuated_indices = [
            i
            for i, joint in enumerate(cfg.actuated_joints)
            if joint not in removed_joints
        ]
        filtered_actuated = [cfg.actuated_joints[i] for i in filtered_actuated_indices]

        filtered_control_modes = [
            cfg.control_modes[i] for i in filtered_actuated_indices
        ]

        filtered_stiffness = (
            [cfg.stiffness[i] for i in filtered_actuated_indices]
            if cfg.stiffness is not None
            else None
        )
        filtered_damping = (
            [cfg.damping[i] for i in filtered_actuated_indices]
            if cfg.damping is not None
            else None
        )
        filtered_joint_damping = (
            [cfg.joint_damping[i] for i in filtered_observed_indices]
            if cfg.joint_damping is not None
            else None
        )
        filtered_joint_frictionloss = (
            [cfg.joint_frictionloss[i] for i in filtered_observed_indices]
            if cfg.joint_frictionloss is not None
            else None
        )
        filtered_armature = (
            [cfg.armature[i] for i in filtered_actuated_indices]
            if cfg.armature is not None
            else None
        )

        return URDFtoMuJoCoLoaderCfg(
            observed_joints=filtered_observed,
            actuated_joints=filtered_actuated,
            control_modes=filtered_control_modes,
            stiffness=filtered_stiffness,
            damping=filtered_damping,
            joint_damping=filtered_joint_damping,
            joint_frictionloss=filtered_joint_frictionloss,
            armature=filtered_armature,
            all_missing_joints_as_sites=cfg.all_missing_joints_as_sites,
            framequat_sensors_cfg=cfg.framequat_sensors_cfg,
            gyro_sensors_cfg=cfg.gyro_sensors_cfg,
            cameras_cfg=cfg.cameras_cfg,
            equality_constraints_cfg=cfg.equality_constraints_cfg,
            ball_joint_damping=cfg.ball_joint_damping,
            ball_joint_armature=cfg.ball_joint_armature,
            ball_joint_frictionloss=cfg.ball_joint_frictionloss,
        )

    @staticmethod
    def load_urdf(urdf_path: str, mesh_path: str, cfg: URDFtoMuJoCoLoaderCfg):
        """
        Load the URDF from the file.

        Args:
            urdf_path (Path): The URDF file path.
            cfg (URDFtoMuJoCoLoaderCfg): The loader configuration.

        Returns:
            URDFtoMuJoCoLoader: The initialized loader instance.
        """
        normalized_cfg = URDFtoMuJoCoLoader._normalize_cfg(cfg)

        urdf_string = URDFtoMuJoCoLoader.simplify_urdf(
            urdf_path,
            normalized_cfg.observed_joints,
            None,
            None,
        )
        reduced_urdf = remove_gazebo_elements(urdf_string)

        # --- Spherical joint handling ---
        # Detect triplets of revolute joints that represent spherical joints
        # (iDynTree convention: 3 revolute joints with x/y/z axes + 2 dummy links)
        spherical_groups = detect_spherical_joint_groups(normalized_cfg.observed_joints)
        ball_joint_map = {}
        if spherical_groups:
            reduced_urdf, ball_joint_map = collapse_spherical_revolute_triplets(
                reduced_urdf, spherical_groups
            )
            print(
                f"Collapsed {len(spherical_groups)} spherical joint group(s) "
                "into ball joint placeholders: "
                f"{[g['base_name'] for g in spherical_groups]}"
            )

        urdf_for_mjcf = add_mujoco_element(reduced_urdf, mesh_path)
        mjcf = load_urdf_into_mjcf(urdf_for_mjcf)
        mjcf = separate_left_right_collision_groups(mjcf)

        # Convert placeholder hinge joints to MuJoCo ball joints
        if ball_joint_map:
            convert_hinge_to_ball_joints(
                mjcf,
                ball_joint_map,
                damping=normalized_cfg.ball_joint_damping,
                armature=normalized_cfg.ball_joint_armature,
                frictionloss=normalized_cfg.ball_joint_frictionloss,
            )

        # Build the set of spherical joint names (all 3 axes) for filtering
        spherical_joint_names = set()
        for group in spherical_groups:
            spherical_joint_names.update(
                [group["joint_x"], group["joint_y"], group["joint_z"]]
            )

        # Ball-joint triplets are passive: remove them from observed + actuated
        # vectors before adding actuators and setting per-joint values.
        mjcf_cfg = URDFtoMuJoCoLoader._filter_cfg_for_removed_joints(
            normalized_cfg, spherical_joint_names
        )

        # Use the reduced URDF (from iDynTree) for computing site transforms.
        # iDynTree modifies joint origins when merging links during model
        # reduction, so the reduced URDF has origins consistent with the MJCF
        # body frames produced by MuJoCo.
        missing_joint_sites = URDFtoMuJoCoLoader.get_missing_joint_sites(
            reduced_urdf,
            mjcf,
            observed_joints=mjcf_cfg.observed_joints,
            all_missing_joints_as_sites=mjcf_cfg.all_missing_joints_as_sites,
        )

        loader = URDFtoMuJoCoLoader(mjcf, mjcf_cfg)
        loader.set_armature(mjcf_cfg.armature)
        loader.add_sites_for_missing_joints(missing_joint_sites)
        loader.add_framequat_sensors(mjcf_cfg.framequat_sensors_cfg)
        loader.add_gyro_sensors(mjcf_cfg.gyro_sensors_cfg)
        loader.add_cameras(mjcf_cfg.cameras_cfg)
        loader.add_equality_constraints(mjcf_cfg.equality_constraints_cfg)
        return loader

    @staticmethod
    def _normalize_framequat_sensor_cfg(
        sensor_cfg: Union[FrameQuatSensorCfg, Dict[str, Any]],
    ) -> FrameQuatSensorCfg:
        if isinstance(sensor_cfg, FrameQuatSensorCfg):
            return sensor_cfg

        if not isinstance(sensor_cfg, dict):
            raise TypeError(
                "Each framequat sensor configuration must be a FrameQuatSensorCfg "
                "or a dict with keys objname, objtype (or obtype), and name."
            )

        objname = sensor_cfg.get("objname")
        objtype = sensor_cfg.get("objtype", sensor_cfg.get("obtype"))
        name = sensor_cfg.get("name")

        if objname is None or objtype is None or name is None:
            raise ValueError(
                "Each framequat sensor requires objname, objtype (or obtype), and name."
            )

        return FrameQuatSensorCfg(objname=objname, objtype=objtype, name=name)

    def add_framequat_sensors(
        self,
        framequat_sensors_cfg: Union[
            None, List[Union[FrameQuatSensorCfg, Dict[str, Any]]]
        ] = None,
    ):
        if framequat_sensors_cfg is None:
            # skip adding sensors if no configuration is provided
            return

        for sensor_cfg in framequat_sensors_cfg:
            normalized_cfg = self._normalize_framequat_sensor_cfg(sensor_cfg)
            add_framequat_sensor(
                self.mjcf,
                objname=normalized_cfg.objname,
                objtype=normalized_cfg.objtype,
                name=normalized_cfg.name,
            )

    @staticmethod
    def _normalize_gyro_sensor_cfg(
        sensor_cfg: Union[GyroSensorCfg, Dict[str, Any]],
    ) -> GyroSensorCfg:
        if isinstance(sensor_cfg, GyroSensorCfg):
            return sensor_cfg

        if not isinstance(sensor_cfg, dict):
            raise TypeError(
                "Each gyro sensor configuration must be a GyroSensorCfg "
                "or a dict with keys site and name."
            )

        site = sensor_cfg.get("site", sensor_cfg.get("objname"))
        name = sensor_cfg.get("name")

        if site is None:
            raise ValueError("Each gyro sensor requires site.")

        return GyroSensorCfg(site=site, name=name)

    def add_gyro_sensors(
        self,
        gyro_sensors_cfg: Union[
            None, List[Union[GyroSensorCfg, Dict[str, Any]]]
        ] = None,
    ):
        if gyro_sensors_cfg is None:
            return

        for sensor_cfg in gyro_sensors_cfg:
            normalized_cfg = self._normalize_gyro_sensor_cfg(sensor_cfg)
            add_gyro_sensor(
                self.mjcf,
                site=normalized_cfg.site,
                name=normalized_cfg.name,
            )

    @staticmethod
    def _normalize_camera_cfg(
        camera_cfg: Union[CameraCfg, Dict[str, Any]],
    ) -> CameraCfg:
        if isinstance(camera_cfg, CameraCfg):
            return camera_cfg

        if not isinstance(camera_cfg, dict):
            raise TypeError(
                "Each camera configuration must be a CameraCfg "
                "or a dict with keys name, site (or link), and fovy."
            )

        name = camera_cfg.get("name")
        site = camera_cfg.get("site", camera_cfg.get("link"))
        fovy = camera_cfg.get("fovy")

        if name is None or site is None:
            raise ValueError(
                "Each camera configuration requires name and site (or link)."
            )

        return CameraCfg(name=name, site=site, fovy=fovy)

    def add_cameras(
        self,
        camera_cfg: Union[None, List[Union[CameraCfg, Dict[str, Any]]]] = None,
    ):
        if camera_cfg is None:
            return

        for camera in camera_cfg:
            normalized_cfg = self._normalize_camera_cfg(camera)
            add_camera_to_site(
                self.mjcf,
                name=normalized_cfg.name,
                site=normalized_cfg.site,
                fovy=normalized_cfg.fovy,
            )

    @staticmethod
    def _normalize_equality_constraint_cfg(
        eq_cfg: Union[EqualityConstraintCfg, Dict[str, Any]],
    ) -> EqualityConstraintCfg:
        if isinstance(eq_cfg, EqualityConstraintCfg):
            return eq_cfg

        if not isinstance(eq_cfg, dict):
            raise TypeError(
                "Each equality constraint configuration must be an "
                "EqualityConstraintCfg or a dict with keys site1 and site2."
            )

        site1 = eq_cfg.get("site1")
        site2 = eq_cfg.get("site2")
        constraint_type = eq_cfg.get("constraint_type", "connect")
        # get solimp and solref if provided, otherwise default to None
        solimp = eq_cfg.get("solimp")
        solref = eq_cfg.get("solref")

        if site1 is None or site2 is None:
            raise ValueError(
                "Each equality constraint configuration requires site1 and site2."
            )

        return EqualityConstraintCfg(
            site1=site1,
            site2=site2,
            constraint_type=constraint_type,
            solimp=solimp,
            solref=solref,
        )

    def add_equality_constraints(
        self,
        equality_constraints_cfg: Union[
            None, List[Union[EqualityConstraintCfg, Dict[str, Any]]]
        ] = None,
    ):
        """Add equality constraints (connect/weld) to the MJCF model.

        Uses the existing ``add_equality_constraints_for_sites`` helper to
        create ``<connect>`` or ``<weld>`` elements inside ``<equality>``.

        Args:
            equality_constraints_cfg: List of ``EqualityConstraintCfg``
                dataclasses or dicts with keys ``site1``, ``site2``, and
                optionally ``constraint_type`` (default ``"connect"``), ``solimp``, and ``solref``.
                If ``None``, no constraints are added.
        """
        if equality_constraints_cfg is None:
            return

        # Group by (constraint_type, solimp, solref) so each call to the helper
        # can pass homogeneous solver parameters.
        by_group: Dict[tuple, List[tuple]] = {}
        for cfg in equality_constraints_cfg:
            normalized = self._normalize_equality_constraint_cfg(cfg)
            group_key = (
                normalized.constraint_type,
                tuple(normalized.solimp) if normalized.solimp is not None else None,
                tuple(normalized.solref) if normalized.solref is not None else None,
            )
            by_group.setdefault(group_key, []).append(
                (normalized.site1, normalized.site2)
            )

        for (constraint_type, solimp, solref), site_pairs in by_group.items():
            add_equality_constraints_for_sites(
                self.mjcf,
                site_pairs,
                constraint_type=constraint_type,
                solimp=list(solimp) if solimp is not None else None,
                solref=list(solref) if solref is not None else None,
            )

    @staticmethod
    def get_missing_joint_sites(
        robot_urdf: ET.Element,
        mjcf: ET.Element,
        observed_joints: List[str] = None,
        all_missing_joints_as_sites: bool = False,
    ) -> List[dict]:
        """Extract metadata for URDF links missing as bodies in MJCF.

        Links can disappear from the MJCF for two reasons:
        1. MuJoCo lumps fixed-joint child bodies into their parents.
        2. iDynTree removes non-fixed joints not in the controlled list during
           model reduction, and their child links are merged too.

        This method identifies all such missing child links and builds site
        descriptors so that their frames can be recovered as MuJoCo sites.

        Args:
            robot_urdf (ET.Element): The URDF root element.
            mjcf (ET.Element): The MJCF root element.
            observed_joints (List[str]): The list of observed joint names.
                Used only for context; detection is based on comparing all URDF
                links against MJCF bodies.
            all_missing_joints_as_sites (bool): If True, enable site generation
                for the missing links.

        Returns:
            List[dict]: Site descriptors with link name, parent/child links and origin.
        """
        if not all_missing_joints_as_sites:
            return []

        if observed_joints is None:
            observed_joints = []

        all_joint_elements = robot_urdf.findall(".//joint")

        # Collect all URDF link names
        urdf_link_names = {
            link.attrib["name"]
            for link in robot_urdf.findall(".//link")
            if "name" in link.attrib
        }

        # Collect MJCF body names
        mjcf_body_names = {
            body.attrib["name"]
            for body in mjcf.findall(".//body")
            if "name" in body.attrib
        }

        # Map each child link to its parent joint (needed for origin + ancestor walk-up)
        joint_by_child_link = {}
        for joint in all_joint_elements:
            child = joint.find("child")
            if child is not None and "link" in child.attrib:
                joint_by_child_link[child.attrib["link"]] = joint

        # Find all URDF links that have no corresponding body in MJCF
        missing_links = {}
        for link_name in urdf_link_names:
            if link_name not in mjcf_body_names:
                joint = joint_by_child_link.get(link_name)
                if joint is not None:
                    missing_links[link_name] = joint

        fixed_joint_sites = []
        for missing_link_name, joint in missing_links.items():

            parent_link = joint.find("parent").attrib["link"]
            child_link = missing_link_name
            origin = joint.find("origin")
            xyz = origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            rpy = origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0"

            # Build all ancestor-link candidates with transform from ancestor link frame
            # to the fixed-joint site frame. This is needed when MuJoCo lumps nested
            # fixed joints and intermediate bodies disappear.
            ancestor_candidates = []
            current_link = child_link
            current_rot_to_site = URDFtoMuJoCoLoader.identity_rot()
            current_pos_to_site = np.zeros(3)

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
                    "name": missing_link_name,
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
        rot = URDFtoMuJoCoLoader.urdf_rpy_to_rot(rpy)
        return URDFtoMuJoCoLoader.rot_to_quat_str(rot)

    @staticmethod
    def identity_rot() -> np.ndarray:
        """Return a 3x3 identity rotation matrix."""
        return np.eye(3)

    @staticmethod
    def str_to_vec(xyz: str) -> np.ndarray:
        """Parse a space-separated string into a numpy array."""
        return np.array(list(map(float, xyz.split())))

    @staticmethod
    def vec_to_str(vec: np.ndarray) -> str:
        """Format a 3-element vector as a space-separated string."""
        return f"{vec[0]} {vec[1]} {vec[2]}"

    @staticmethod
    def compose_rot(r_ab: np.ndarray, r_bc: np.ndarray) -> np.ndarray:
        """Compose two rotation matrices: R_ac = R_ab @ R_bc."""
        return r_ab @ r_bc

    @staticmethod
    def compose_pos(
        r_ab: np.ndarray,
        p_ab: np.ndarray,
        p_bc: np.ndarray,
    ) -> np.ndarray:
        """Compose positions: p_ac = p_ab + R_ab @ p_bc."""
        return p_ab + r_ab @ p_bc

    @staticmethod
    def urdf_rpy_to_rot(rpy: str) -> np.ndarray:
        """URDF RPY (fixed-axis / extrinsic XYZ) to 3x3 rotation matrix via scipy."""
        angles = list(map(float, rpy.split()))  # [roll, pitch, yaw]
        return Rotation.from_euler("xyz", angles, degrees=False).as_matrix()

    @staticmethod
    def rot_to_quat_str(rot: np.ndarray) -> str:
        """Convert a 3x3 rotation matrix to a MuJoCo quaternion string (w x y z)."""
        quat = URDFtoMuJoCoLoader.rot_to_quat(rot)
        return f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}"

    @staticmethod
    def rot_to_quat(rot: np.ndarray) -> np.ndarray:
        """Convert a 3x3 rotation matrix to quaternion [w, x, y, z] (MuJoCo convention)."""
        # scipy returns [x, y, z, w]
        xyzw = Rotation.from_matrix(np.asarray(rot)).as_quat()
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

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
            raise ValueError(
                f"Error loading the URDF model from {urdf_path}. Check the file path or if the joints are correct."
            )
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

    def set_armature(self, armature: Union[None, List[float]]):
        """Set the armature attribute on each actuated joint in the MJCF model.

        Args:
            armature (List[float] | None): Armature values, one per actuated joint.
                If None, no armature attribute is set.
        """
        if armature is None:
            return

        if len(armature) != len(self.actuated_joints):
            raise ValueError(
                f"Length of armature ({len(armature)}) must match "
                f"the number of actuated_joints ({len(self.actuated_joints)})."
            )

        for joint_name, value in zip(self.actuated_joints, armature):
            joint_elem = self.mjcf.find(f".//joint[@name='{joint_name}']")
            if joint_elem is not None:
                joint_elem.set("armature", str(value))

    def set_observed_joint_dynamics(
        self,
        damping: Union[None, List[float]] = None,
        frictionloss: Union[None, List[float]] = None,
    ):
        """Set damping/frictionloss attributes on observed joints.

        Args:
            damping (List[float] | None): Per-observed-joint damping values.
            frictionloss (List[float] | None): Per-observed-joint frictionloss values.
        """
        if damping is None and frictionloss is None:
            return

        if damping is not None and len(damping) != len(self.observed_joints):
            raise ValueError(
                f"Length of joint_damping ({len(damping)}) must match "
                f"the number of observed_joints ({len(self.observed_joints)})."
            )

        if frictionloss is not None and len(frictionloss) != len(self.observed_joints):
            raise ValueError(
                f"Length of joint_frictionloss ({len(frictionloss)}) must match "
                f"the number of observed_joints ({len(self.observed_joints)})."
            )

        for index, joint_name in enumerate(self.observed_joints):
            joint_elem = self.mjcf.find(f".//joint[@name='{joint_name}']")
            if joint_elem is None:
                continue
            if damping is not None:
                joint_elem.set("damping", str(float(damping[index])))
            if frictionloss is not None:
                joint_elem.set("frictionloss", str(float(frictionloss[index])))

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

    def add_actuator(
        self,
        joint: str,
        control_mode: ControlMode,
        stiffness: Union[None, float] = None,
        damping: Union[None, float] = None,
    ):
        """
        Add an actuator to the MJCF model.

        Args:
            joint (str): The joint name.
            control_mode (ControlMode): The control mode.
            stiffness (float | None): Position gain used as kp for position mode.
            damping (float | None): Joint damping applied for position mode.
        """
        if control_mode == ControlMode.POSITION:
            if stiffness is None:
                raise ValueError(
                    f"POSITION control for joint {joint} requires stiffness (kp)."
                )
            if damping is None:
                raise ValueError(
                    f"POSITION control for joint {joint} requires damping."
                )

            joint_mjcf = self.mjcf.find(f".//joint[@name='{joint}']")
            if joint_mjcf is None:
                raise ValueError(f"Joint {joint} not found in the MJCF model.")
            if "range" not in joint_mjcf.attrib:
                raise ValueError(
                    f"Joint {joint} is missing range, required for position control."
                )

            ctrlrange = list(map(float, joint_mjcf.attrib["range"].split()))
            add_position_actuator(
                self.mjcf,
                joint=joint,
                ctrlrange=ctrlrange,
                kp=float(stiffness),
            )
            joint_mjcf.set("damping", str(float(damping)))
        elif control_mode == ControlMode.TORQUE:
            add_torque_actuator(self.mjcf, joint=joint, ctrlrange=None)
        elif control_mode == ControlMode.VELOCITY:
            raise NotImplementedError("Velocity control is not implemented yet.")
        else:
            raise ValueError("Control mode not recognized.")

    def set_actuated_joints(
        self,
        joints: List[str],
        stiffness: Union[None, List[float]] = None,
        damping: Union[None, List[float]] = None,
    ):
        """Set actuated joints and add actuators according to per-joint modes."""
        self.actuated_joints = list(joints)
        joint_elements = {
            joint.attrib["name"]: joint for joint in self.mjcf.findall(".//joint")
        }

        if stiffness is not None and len(stiffness) != len(self.actuated_joints):
            raise ValueError(
                f"Length of stiffness ({len(stiffness)}) must match "
                f"the number of actuated_joints ({len(self.actuated_joints)})."
            )

        if damping is not None and len(damping) != len(self.actuated_joints):
            raise ValueError(
                f"Length of damping ({len(damping)}) must match "
                f"the number of actuated_joints ({len(self.actuated_joints)})."
            )

        for i, actuated_joint in enumerate(self.actuated_joints):
            joint_element = joint_elements.get(actuated_joint)
            if joint_element is None:
                raise ValueError(f"Joint {actuated_joint} not found in the MJCF model.")

            if actuated_joint not in self.control_mode:
                self.control_mode[actuated_joint] = ControlMode.TORQUE

            self.add_actuator(
                actuated_joint,
                self.control_mode[actuated_joint],
                stiffness=(stiffness[i] if stiffness is not None else None),
                damping=(damping[i] if damping is not None else None),
            )

    def add_sites_for_missing_joints(self, fixed_joint_sites: List[dict]):
        """Add one MJCF site for each configured missing URDF joint.

        The site is attached to the fixed joint child link body when available.
        If the child body is not present in MJCF, the site is attached to the
        parent link body at the URDF joint origin.
        """
        for fixed_joint_site in fixed_joint_sites:
            site_name = fixed_joint_site["child_link"]

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
            mjcf = ET.fromstring(
                ET.tostring(self.mjcf, encoding="unicode", method="xml")
            )
            ET.indent(mjcf, space="  ")
        return ET.tostring(mjcf, encoding="unicode", method="xml")

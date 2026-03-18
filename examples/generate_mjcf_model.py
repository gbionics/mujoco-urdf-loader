import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import resolve_robotics_uri_py as rru
import tomli
from mujoco_urdf_loader import (
    ControlMode,
    MujocoWrapper,
    URDFtoMuJoCoLoader,
    URDFtoMuJoCoLoaderCfg,
)
from mujoco_urdf_loader.mjcf_fcn import (
    add_sites_for_fixed_joints,
    add_equality_constraints_for_sites,
    convert_spherical_joints_to_ball,
)
from mujoco_urdf_loader.urdf_fcn import (
    get_mesh_path,
    find_spherical_revolute_joints_in_urdf,
    set_min_mass_for_zero_mass_links,
)

# from gb_gene_models.paths import get_share_dir


def load_config(config_path):
    """Load robot configuration from TOML file."""
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    robot_config = config["robot"]
    robot_name = robot_config.get("name", "lowerbodyrsu")
    controlled_joints = robot_config["controlled_joints"]

    # Parse control mode
    control_mode_str = robot_config.get("control_mode", "TORQUE").upper()
    control_mode = getattr(ControlMode, control_mode_str)
    control_modes = [control_mode] * len(controlled_joints)

    stiffness = robot_config["stiffness"]
    damping = robot_config["damping"]

    # check lengths
    # if not (len(controlled_joints) == len(stiffness) == len(damping)):
    #     raise ValueError(
    #         "Length of controlled_joints, stiffness, and damping must be the same."
    #     )

    # Print loaded configuration
    print(f"Loaded robot configuration from {config_path}:")
    print(f"  Robot name: {robot_name}")
    print(f"  Controlled joints: {controlled_joints}")
    print(f"  Control mode: {control_modes[0]}")
    print("  Stiffness and Damping:")
    for joint, k, d in zip(controlled_joints, stiffness, damping):
        print(f"    {joint}: stiffness= {k} , damping = {d}")

    return robot_name, controlled_joints, control_modes, stiffness, damping


def main():
    parser = argparse.ArgumentParser(description="Generate MuJoCo XML model from URDF")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "robot_config.toml",
        help="Path to robot configuration TOML file (default: config/robot_config.toml)",
    )
    args = parser.parse_args()

    # Load configuration
    robot_name, controlled_joints, control_modes, stiffness, damping = load_config(
        args.config
    )
    cfg = URDFtoMuJoCoLoaderCfg(controlled_joints, control_modes, stiffness, damping)

    # Resolve URDF path using robotics URI
    # os.environ["GZ_SIM_RESOURCE_PATH"] = str(get_share_dir())
    # urdf_path = str(
    #     rru.resolve_robotics_uri(f"package://gene/robots/{robot_name}/model.urdf")
    # )
    urdf_path = r"C:\Users\PunithReddyVanteddu\Documents\Github\gb-lowerbody-models\mjcf\lowerbody\lowerbodyqddrsu\model.urdf"
    print(f"Resolved URDF path: {urdf_path}")
    mesh_path = r"C:\Users\PunithReddyVanteddu\Documents\Github\gb-lowerbody-models\urdf\lowerbody\meshes\simmechanics"

    # Parse the URDF to detect spherical revolute joints (3-rev chains with
    # zero-mass intermediate links).  These must be included in the joints
    # list passed to iDynTree so the rod bodies survive simplification.
    robot_urdf = ET.parse(urdf_path).getroot()
    spherical_joints = find_spherical_revolute_joints_in_urdf(robot_urdf)
    if spherical_joints:
        print(
            f"  Detected {len(spherical_joints)} spherical revolute joints: {spherical_joints}"
        )

    # Build the MJCF manually so we can include spherical joints in
    # iDynTree simplification without adding actuators for them.
    from mujoco_urdf_loader.urdf_fcn import remove_gazebo_elements, add_mujoco_element
    from mujoco_urdf_loader.generator import load_urdf_into_mjcf
    from mujoco_urdf_loader.mjcf_fcn import separate_left_right_collision_groups

    all_joints = controlled_joints + spherical_joints
    all_stiffness = stiffness + [0.0] * len(spherical_joints) if stiffness else None
    all_damping = damping + [0.0] * len(spherical_joints) if damping else None

    urdf_simplified = URDFtoMuJoCoLoader.simplify_urdf(
        urdf_path, all_joints, all_stiffness, all_damping
    )
    urdf_simplified = remove_gazebo_elements(urdf_simplified)
    set_min_mass_for_zero_mass_links(urdf_simplified)
    urdf_simplified = add_mujoco_element(urdf_simplified, mesh_path)
    mjcf = load_urdf_into_mjcf(urdf_simplified)
    mjcf = separate_left_right_collision_groups(mjcf)

    # Convert 3-revolute spherical joint chains to MuJoCo ball joints
    convert_spherical_joints_to_ball(mjcf)

    # Create the loader with only the original controlled joints (no actuators
    # for ball joints).
    cfg = URDFtoMuJoCoLoaderCfg(controlled_joints, control_modes, stiffness, damping)
    loader = URDFtoMuJoCoLoader(mjcf, cfg)

    # Add sites for fixed joints (sensor frames, etc.)
    add_sites_for_fixed_joints(loader.mjcf, robot_urdf)

    # Add equality constraints between site pairs
    add_equality_constraints_for_sites(
        loader.mjcf,
        [
            ("r_anklecr_in_fixed_joint", "r_rod_in_bottom_fixed_joint"),
            ("l_anklecr_in_fixed_joint", "l_rod_in_bottom_fixed_joint"),
            ("r_anklecr_out_fixed_joint", "r_rod_out_bottom_fixed_joint"),
            ("l_anklecr_out_fixed_joint", "l_rod_out_bottom_fixed_joint"),
        ],
        constraint_type="connect",  # or "weld" for rigid connection
    )
    # add_sites_for_ft(loader.mjcf, robot_urdf)
    # add_sites_for_imu(loader.mjcf, robot_urdf)

    # Note: add_sites_to_body requires specific parent_body and target_link parameters
    # Example: add_sites_to_body(loader.mjcf, robot_urdf, "parent_body_name", "target_link_name")

    # save xml_str to a file
    xml_path = Path(urdf_path).parent / f"model.xml"
    with open(xml_path, "w") as f:
        f.write(loader.get_mjcf_string())
    print(f"MuJoCo XML model saved to: {xml_path}")

    # include the model in a simple world
    world_str = f"""
        <mujoco model="{robot_name}World">
            <include file="{xml_path}"/>

            <visual>
                <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
                <rgba haze="0.15 0.25 0.35 1"/>
                <global azimuth="120" elevation="-20"/>
            </visual>

            <asset>
                <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
                <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
                markrgb="0.8 0.8 0.8" width="300" height="300"/>
                <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
            </asset>

            <worldbody>
                <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
                <camera name="default" pos="0.846 -1.465 0.916" xyaxes="0.866 0.500 0.000 -0.171 0.296 0.940"/>
                <geom name="floor" pos="0 0 0" size="0 0 0.05" type="plane" material="groundplane"/>
            </worldbody>
        </mujoco>
        """

    mujoco_wrapper = MujocoWrapper(world_str)
    model = mujoco_wrapper.model
    data = mujoco_wrapper.data

    # smoke test: load model in Mujoco
    model = mujoco.MjModel.from_xml_string(world_str)
    data = mujoco.MjData(model)

    # Set base pose: position and orientation
    data.qpos[0] = 0.0  # x position
    data.qpos[1] = 0.0  # y position
    data.qpos[2] = 1.5  # z position - lift the robot above the ground
    data.qpos[3] = 1.0  # quaternion w (identity rotation)
    data.qpos[4] = 0.0  # quaternion x
    data.qpos[5] = 0.0  # quaternion y
    data.qpos[6] = 0.0  # quaternion z

    # visualize the model
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()

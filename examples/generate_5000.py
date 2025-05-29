import os
import tempfile
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
import idyntree.swig as idyntree
import numpy as np
from mujoco_urdf_loader.mjcf_fcn import (
    add_camera,
    add_new_worldbody,
    add_position_actuator,
    separate_left_right_collision_groups,
    set_joint_damping,
    add_sites_for_ft,
    add_sites_for_imu,
    add_sites_to_body,
    add_sensors_to_sites,
)
from mujoco_urdf_loader.urdf_fcn import (
    add_mujoco_element,
    get_robot_urdf,
    remove_gazebo_elements,
)
import mujoco_urdf_loader.loader as loader


# # add sites for turbines
def add_sites_turbines(
    mjcf: ET.Element, body_name: str, geom_mesh: str, name: str = None
) -> ET.Element:
    """Add sites to specific bodies in the mjcf file.

    Args:
        mjcf (ET.Element): The mjcf file as ElementTree.
        body_name (str): The name of the body to add the site to.
        geom_type (str): The type of the geom to add the site to.
        name (str): The name of the site (default: f"{geom_type}_site").

    Returns:
        ET.Element: The modified mjcf file.
    """
    for body in mjcf.findall(f".//body[@name='{body_name}']"):
        # Iterate through <geom> elements within the specific <body>
        for geom in body.findall("geom"):
            if geom.attrib.get("mesh") == geom_mesh:
                geom_pos = geom.attrib.get("pos", "")
                geom_quat = geom.attrib.get("quat", "")

                # Create <site> element
                site = ET.SubElement(body, "site")
                site.set(
                    "name",
                    name if name is not None else f"{geom.attrib.get('mesh')}_site",
                )
                site.set("pos", geom_pos)
                site.set("quat", geom_quat)

    return mjcf


def add_jet_turbine_motors(mjcf: ET.Element) -> ET.Element:
    """
    Add motor actuators for jet turbines to the MJCF file.

    Args:
        mjcf (ET.Element): The MJCF file as ElementTree.

    Returns:
        ET.Element: The modified MJCF file with added motors.
    """
    # Find or create the <actuator> element
    actuator = mjcf.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(mjcf, "actuator")

    # Define the motors to add
    motors = [
        {
            "gear": "0 0 -1 0 0 0",
            "site": "l_arm_turbine",
            "name": "l_arm_jet_turbine",
            "ctrlrange": "0 250",
        },
        {
            "gear": "0 0 -1 0 0 0",
            "site": "r_arm_turbine",
            "name": "r_arm_jet_turbine",
            "ctrlrange": "0 250",
        },
        {
            "gear": "0 0 -1 0 0 0",
            "site": "l_jet_turbine",
            "name": "chest_l_jet_turbine",
            "ctrlrange": "0 250",
        },
        {
            "gear": "0 0 -1 0 0 0",
            "site": "r_jet_turbine",
            "name": "chest_r_jet_turbine",
            "ctrlrange": "0 250",
        },
    ]

    # Add each motor to the <actuator> block
    for motor in motors:
        motor_elem = ET.SubElement(actuator, "motor")
        motor_elem.set("gear", motor["gear"])
        motor_elem.set("site", motor["site"])
        motor_elem.set("name", motor["name"])
        motor_elem.set("ctrlrange", motor["ctrlrange"])

    return mjcf


def add_ft_sites_to_chest(
    mjcf: ET.Element, urdf: ET.Element, parent_body: str
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
        if "r_jet_ft" in child_link and parent_link == "chest":
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


# Function to format XML
def format_xml(mjcf: ET.Element) -> str:
    """
    Formats an XML ElementTree into a human-readable string with proper indentation.

    Args:
        mjcf (ET.Element): The XML ElementTree.

    Returns:
        str: The formatted XML string.
    """
    from xml.dom import minidom

    xml_pretty = minidom.parseString(ET.tostring(mjcf)).toprettyxml(indent="    ")
    return "\n".join(line for line in xml_pretty.splitlines() if line.strip())


def prepend_mujoco_configuration(mjcf: ET.Element) -> ET.Element:
    """
    Prepend the specified MuJoCo configuration to the MJCF model.

    Args:
        mjcf (ET.Element): The MJCF file as ElementTree.

    Returns:
        ET.Element: The modified MJCF file with the configuration prepended.
    """
    # Create a new root element with the desired configuration
    new_root = ET.Element("mujoco", attrib={"model": "iCub"})

    # Add the <visual> element
    visual = ET.SubElement(new_root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        diffuse="0.6 0.6 0.6",
        ambient="0.3 0.3 0.3",
        specular="0 0 0",
    )
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    ET.SubElement(visual, "global", azimuth="120", elevation="-20")

    # Add the <asset> element
    asset = ET.SubElement(new_root, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="2d",
        name="groundplane",
        builtin="checker",
        mark="edge",
        rgb1="0.2 0.3 0.4",
        rgb2="0.1 0.2 0.3",
        markrgb="0.8 0.8 0.8",
        width="300",
        height="300",
    )
    ET.SubElement(
        asset,
        "material",
        name="groundplane",
        texture="groundplane",
        texuniform="true",
        texrepeat="5 5",
    )

    # Add the <worldbody> element
    worldbody = ET.SubElement(new_root, "worldbody")
    ET.SubElement(
        worldbody,
        "geom",
        name="ground",
        type="plane",
        pos="0 0 0",
        size="0 0 0.1",
        material="groundplane",
    )
    ET.SubElement(
        worldbody,
        "light",
        name="main_light",
        pos="0 0 2",
        dir="0 0 -1",
        diffuse="1 1 1",
        specular="0.5 0.5 0.5",
        directional="true",
    )
    ET.SubElement(worldbody, "camera", name="fixed", pos="1 -1 1", xyaxes="1 0 0 0 1 0")
    ET.SubElement(
        worldbody,
        "camera",
        name="isometric",
        mode="targetbodycom",
        target="chest",
        pos="2 -2 2",
        xyaxes="1 0 0 0 0 1",
    )

    # Add the <option> element
    ET.SubElement(
        new_root, "option", gravity="0 0 -9.81", iterations="50", cone="elliptic"
    )

    # Add the <compiler> element
    ET.SubElement(new_root, "compiler", angle="radian")

    # Append the existing MJCF content to the new root
    for child in mjcf:
        new_root.append(child)

    return new_root


# Function to process a single URDF file
def process_urdf_to_mjcf(urdf_path, output_path):
    # Process the URDF dynamically
    robot_urdf = get_robot_urdf(urdf_path)
    robot_urdf = remove_gazebo_elements(robot_urdf)
    mesh_path = os.path.dirname(urdf_path)

    # Create the MJCF configuration
    controlled_joints = [
        "l_hip_pitch",
        "l_hip_roll",
        "l_hip_yaw",
        "l_knee",
        "l_ankle_pitch",
        "l_ankle_roll",
        "r_hip_pitch",
        "r_hip_roll",
        "r_hip_yaw",
        "r_knee",
        "r_ankle_pitch",
        "r_ankle_roll",
        "torso_roll",
        "torso_pitch",
        "torso_yaw",
        "l_shoulder_pitch",
        "l_shoulder_roll",
        "l_shoulder_yaw",
        "l_elbow",
        "r_shoulder_pitch",
        "r_shoulder_roll",
        "r_shoulder_yaw",
        "r_elbow",
        "neck_pitch",
        "neck_roll",
        "neck_yaw",
    ]
    control_modes = [loader.ControlMode.POSITION] * len(controlled_joints)
    stiffness = [0.0] * len(controlled_joints)
    damping = [0.0] * len(controlled_joints)

    # Create the configuration object
    cfg = loader.URDFtoMuJoCoLoaderCfg(
        controlled_joints=controlled_joints,
        control_modes=control_modes,
        stiffness=stiffness,
        damping=damping,
    )

    # Load the URDF into MJCF
    loader_instance = loader.URDFtoMuJoCoLoader.load_urdf(
        urdf_path, mesh_path, cfg  # Pass the correct cfg object
    )
    mjcf = loader_instance.mjcf

    # Prepend MuJoCo configuration
    mjcf = prepend_mujoco_configuration(mjcf)

    # Add additional configurations
    add_sites_turbines(mjcf, "chest", "sim_sea_l_jet_turbine", "l_jet_turbine")
    add_sites_turbines(mjcf, "chest", "sim_sea_r_jet_turbine", "r_jet_turbine")
    add_sites_turbines(mjcf, "l_elbow_1", "sim_sea_l_arm_p250", "l_arm_turbine")
    add_sites_turbines(mjcf, "r_elbow_1", "sim_sea_r_arm_p250", "r_arm_turbine")
    add_jet_turbine_motors(mjcf)
    add_sites_for_ft(mjcf, robot_urdf)
    add_ft_sites_to_chest(mjcf, robot_urdf, "chest")
    add_sites_for_imu(mjcf, robot_urdf)
    add_sites_to_body(mjcf, robot_urdf, "l_ankle_2", "l_foot_rear")
    add_sites_to_body(mjcf, robot_urdf, "r_ankle_2", "r_foot_rear")
    add_sites_to_body(mjcf, robot_urdf, "l_ankle_2", "l_foot_front")
    add_sites_to_body(mjcf, robot_urdf, "r_ankle_2", "r_foot_front")
    add_sensors_to_sites(mjcf)
    separate_left_right_collision_groups(mjcf)

    # Save the MJCF to the output path
    formatted_xml = format_xml(mjcf)
    with open(output_path, "w") as f:
        f.write(formatted_xml)

    print(f"Processed {urdf_path} -> {output_path}")


# Main function to process all URDF files in a folder
def process_all_urdfs(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through URDF files in the input folder
    for i in range(
        1, 5001
    ):  # Assuming files are named 001.urdf, 002.urdf, ..., 5000.urdf
        urdf_filename = (
            f"{i:03}.urdf"  # Format the number with leading zeros (e.g., 001, 002, ...)
        )
        urdf_path = os.path.join(input_folder, urdf_filename)
        output_filename = f"{i:03}.xml"  # Match the output filename format
        output_path = os.path.join(output_folder, output_filename)

        # Check if the URDF file exists
        if os.path.exists(urdf_path):
            process_urdf_to_mjcf(urdf_path, output_path)
        else:
            print(f"URDF file {urdf_path} not found. Skipping.")


# Example usage
input_folder = "C:/Users/pvanteddu/Documents/GitHub/element_co-generative-cad/Mode_frontierAnalayses/Whole_body_rl/URDF's/5000_models/"  # Replace with the path to your URDF folder
output_folder = "C:/Users/pvanteddu/Documents/GitHub/element_co-generative-cad/Mode_frontierAnalayses/Whole_body_rl/URDF's/5000_models/"  # Replace with the path to your output folder

process_all_urdfs(input_folder, output_folder)

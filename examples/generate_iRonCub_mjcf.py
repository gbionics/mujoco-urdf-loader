import tempfile
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import os
import subprocess
import idyntree.swig as idyntree
import mujoco_urdf_loader.generator as generator
import mujoco_urdf_loader.loader as loader

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
    get_mesh_path,
    get_robot_urdf,
    remove_gazebo_elements,
)


# Print the value of the environment variable
package = os.getenv("IRONCUB_COMPONENT_SOURCE_DIR")
# package = "C:/Users/pvanteddu/Documents/iRonCub_ws/src/component_ironcub/"
# # Load the robot urdf
robot_relative_path = "models/iRonCub-Mk3/iRonCub/robots/iRonCub-Mk3/model.urdf"
mujoco_relative_path = "models/iRonCub-Mk3/iRonCub/robots/iRonCub-Mk3_Mujoco"

# Choose mesh_relative_path based on robot_relative_path
if robot_relative_path.endswith("model_stl.urdf"):
    mesh_relative_path = "../../meshes/stl"
else:
    mesh_relative_path = "../../meshes/obj"


robot_path = os.path.join(
    package, robot_relative_path
)  # "C:/Users/pvanteddu/Documents/iRonCub_ws/src/component_ironcub/models/iRonCub-Mk3/iRonCub/robots/iRonCub-Mk3/model_stl.urdf"
robot_urdf = ET.parse(robot_path).getroot()
robot_urdf = get_robot_urdf(robot_path)
mujoco_path = os.path.join(package, mujoco_relative_path)


mesh_path = os.path.normpath(
    os.path.join(os.path.dirname(robot_path), mesh_relative_path)
)


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

cfg = loader.URDFtoMuJoCoLoaderCfg(controlled_joints, control_modes, stiffness, damping)
mjcf_root = ET.Element("mujoco", attrib={"model": "icub"})
compiler = ET.SubElement(mjcf_root, "compiler", attrib={"balanceinertia": "true"})

loader = loader.URDFtoMuJoCoLoader.load_urdf(robot_path, mesh_path, cfg)
mjcf = loader.mjcf


# Set the damping
set_joint_damping(mjcf, damping=2)


# # add sites for turbines
def add_sites_turbines(
    mjcf: ET.Element,
    body_name: str,
    geom_mesh: str,
    site_name: str = None,
    add_body_name: str = None,
    add_geom_name: str = None,
) -> ET.Element:
    """Add sites to specific bodies in the mjcf file.

    Args:
        mjcf (ET.Element): The mjcf file as ElementTree.
        body_name (str): The name of the body to add the site to.
        geom_type (str): The type of the geom to add the site to.
        site_name (str): The name of the site.
        add_body_name (str): The name of the body to add under the site.
        add_geom_name (str): The name of the geom to add under the site.



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
                    (
                        site_name
                        if site_name is not None
                        else f"{geom.attrib.get('mesh')}_site"
                    ),
                )
                site.set("pos", geom_pos)
                site.set("quat", geom_quat)
                # Create a body
                if add_body_name is not None:
                    add_body = ET.SubElement(
                        body, "body", name=add_body_name, pos=geom_pos, quat=geom_quat
                    )
                    # Add inertial properties to the new body
                    ET.SubElement(
                        add_body,
                        "inertial",
                        {
                            "pos": "0 0 0",
                            "quat": "1 0 0 0",
                            "mass": "0.0",
                            "diaginertia": "0.0 0.0 0.0",
                        },
                    )
                    # Create a geom under the body
                    ET.SubElement(
                        add_body,
                        "geom",
                        name=add_geom_name if add_geom_name else f"{site_name}_geom",
                        type="cylinder",
                        pos="0 0 0.075",
                        quat="1 0 0 0",
                        size="0.02 0.15",
                        rgba="1 0 0 0.5",
                    )

    return mjcf


add_sites_turbines(
    mjcf,
    "chest",
    "sim_sea_l_jet_turbine",
    "l_jet_turbine",
    "l_jet_body",
    "l_jet_cylinder",
)
add_sites_turbines(
    mjcf,
    "chest",
    "sim_sea_r_jet_turbine",
    "r_jet_turbine",
    "r_jet_body",
    "r_jet_cylinder",
)
add_sites_turbines(
    mjcf,
    "l_elbow_1",
    "sim_sea_l_arm_p250",
    "l_arm_turbine",
    "l_arm_body",
    "l_arm_cylinder",
)
add_sites_turbines(
    mjcf,
    "r_elbow_1",
    "sim_sea_r_arm_p250",
    "r_arm_turbine",
    "r_arm_body",
    "r_arm_cylinder",
)


# add sites for the  ft
add_sites_for_ft(mjcf, robot_urdf)


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


# Actuator configuration for each joint
joint_actuator_params = {
    "l_hip_pitch": {"kp": 1200, "dampratio": 0.2, "ctrlrange": (-0.785398, 2.00713)},
    "l_hip_roll": {"kp": 1200, "dampratio": 0.2, "ctrlrange": (-0.174533, 1.95477)},
    "l_hip_yaw": {"kp": 1200, "dampratio": 0.5, "ctrlrange": (-1.39626, 1.39626)},
    "l_knee": {"kp": 1800, "dampratio": 0.8, "ctrlrange": (-1.22173, 0.0872665)},
    "l_ankle_pitch": {"kp": 1800, "dampratio": 0.8, "ctrlrange": (-0.785398, 0.785398)},
    "l_ankle_roll": {"kp": 1800, "dampratio": 0.8, "ctrlrange": (-0.436332, 0.436332)},
    "r_hip_pitch": {"kp": 1200, "dampratio": 0.2, "ctrlrange": (-0.785398, 2.00713)},
    "r_hip_roll": {"kp": 1200, "dampratio": 0.2, "ctrlrange": (-0.174533, 1.95477)},
    "r_hip_yaw": {"kp": 1200, "dampratio": 0.5, "ctrlrange": (-1.39626, 1.39626)},
    "r_knee": {"kp": 1800, "dampratio": 0.8, "ctrlrange": (-1.22173, 0.0872665)},
    "r_ankle_pitch": {"kp": 1800, "dampratio": 0.8, "ctrlrange": (-0.785398, 0.785398)},
    "r_ankle_roll": {"kp": 1800, "dampratio": 0.8, "ctrlrange": (-0.436332, 0.436332)},
    "torso_roll": {"kp": 1500, "dampratio": 0.2, "ctrlrange": (-0.401426, 0.401426)},
    "torso_pitch": {"kp": 1500, "dampratio": 0.3, "ctrlrange": (-0.314159, 0.785398)},
    "torso_yaw": {"kp": 1500, "dampratio": 0.4, "ctrlrange": (-0.750492, 0.750492)},
    "l_shoulder_pitch": {
        "kp": 3000,
        "dampratio": 0.3,
        "ctrlrange": (-3.05433, 0.349066),
    },
    "l_shoulder_roll": {"kp": 3000, "dampratio": 0.3, "ctrlrange": (0.20944, 2.84489)},
    "l_shoulder_yaw": {"kp": 3000, "dampratio": 0.3, "ctrlrange": (-0.872665, 1.39626)},
    "l_elbow": {"kp": 3000, "dampratio": 0.3, "ctrlrange": (-0.0523599, 1.309)},
    "r_shoulder_pitch": {
        "kp": 3000,
        "dampratio": 0.3,
        "ctrlrange": (-3.05433, 0.349066),
    },
    "r_shoulder_roll": {"kp": 3000, "dampratio": 0.3, "ctrlrange": (0.20944, 2.84489)},
    "r_shoulder_yaw": {"kp": 3000, "dampratio": 0.3, "ctrlrange": (-0.872665, 1.39626)},
    "r_elbow": {"kp": 3000, "dampratio": 0.3, "ctrlrange": (-0.0523599, 1.309)},
    "neck_pitch": {"kp": 2000, "dampratio": 0.12, "ctrlrange": (-0.523599, 0.383972)},
    "neck_roll": {"kp": 2000, "dampratio": 0.12, "ctrlrange": (-0.349066, 0.349066)},
    "neck_yaw": {"kp": 2000, "dampratio": 0.12, "ctrlrange": (-0.785398, 0.785398)},
}


def set_position_actuator_params(mjcf, joint_actuator_params):
    """
    Set kp, dampratio, and ctrlrange for position actuators in the MJCF tree.
    """
    for actuator in mjcf.findall(".//actuator/position"):
        joint = actuator.get("joint")
        if joint in joint_actuator_params:
            params = joint_actuator_params[joint]
            actuator.set("kp", str(params["kp"]))
            actuator.set("dampratio", str(params["dampratio"]))
            actuator.set(
                "ctrlrange", f"{params['ctrlrange'][0]} {params['ctrlrange'][1]}"
            )


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


def add_sites_links(
    mjcf: ET.Element, urdf: ET.Element, parent_body: str, site_name: str
) -> ET.Element:
    """
    Add a site with a specified name and pos="0 0 0" to the given body in the MJCF.

    Args:
        mjcf (ET.Element): The MJCF file as ElementTree.
        urdf (ET.Element): The URDF file as ElementTree (not used here).
        parent_body (str): The name of the parent body in the MJCF.
        site_name (str): The name of the site to add.

    Returns:
        ET.Element: The modified MJCF file.
    """
    # Find the parent body in the MJCF
    body = mjcf.find(f".//body[@name='{parent_body}']")
    if body is None:
        print(f"Body {parent_body} not found in MJCF")
        return mjcf

    # Create the new site with pos="0 0 0"
    site = ET.SubElement(body, "site")
    site.set("name", site_name)
    site.set("pos", "0 0 0")

    return mjcf


# Add sites for the links
mjcf = add_sites_links(mjcf, robot_urdf, "chest", "chest_sensor")
mjcf = add_sites_links(mjcf, robot_urdf, "root_link", "root_link_site")

# remove the gazebo elements
robot_urdf = remove_gazebo_elements(robot_urdf)


def add_materials_to_asset(mjcf: ET.Element, materials: list):
    """
    Add material definitions to the <asset> section of the MJCF.

    Args:
        mjcf (ET.Element): The MJCF root element.
        materials (list): List of dicts, each with 'name' and 'rgba' keys.
    """
    asset = mjcf.find("asset")
    if asset is None:
        asset = ET.SubElement(mjcf, "asset")
    for mat in materials:
        ET.SubElement(asset, "material", name=mat["name"], rgba=mat["rgba"])


def assign_materials_to_geoms(mjcf: ET.Element, geom_material_map: dict):
    """
    Assign specified materials to geoms by mesh name.

    Args:
        mjcf (ET.Element): The MJCF root element.
        geom_material_map (dict): Mapping from mesh name to material name.
    """
    for mesh_name, material_name in geom_material_map.items():
        for geom in mjcf.findall(f".//geom[@mesh='{mesh_name}']"):
            geom.set("material", material_name)


# Example usage:
materials = [
    {
        "name": "robot_covers_cad",
        "rgba": "0 0.5 0.5 1",
        "geoms": [
            "sim_icub3_l_ankle_2",
            "sim_icub3_r_ankle_2",
            "sim_icub3_l_hip_3",
            "sim_icub3_r_hip_3",
            "sim_icub3_l_lower_leg",
            "sim_icub3_r_lower_leg",
            "sim_icub3_chest",
            "sim_icub3_l_upperarm",
            "sim_icub3_r_upperarm",
        ],
    },
    {"name": "head_mat", "rgba": "0.8 0.6 0.4 1", "geoms": ["sim_head_head"]},
    {
        "name": "turbine_mat",
        "rgba": "0.1 0.1 0.1 1",
        "geoms": [
            "sim_sea_l_jet_turbine",
            "sim_sea_r_jet_turbine",
            "sim_sea_l_arm_p250",
            "sim_sea_r_arm_p250",
        ],
    },
]

# Add materials to asset
add_materials_to_asset(mjcf, materials)

# Build geom-to-material mapping from the materials list
geom_material_map = {}
for mat in materials:
    for geom_name in mat.get("geoms", []):
        geom_material_map[geom_name] = mat["name"]

assign_materials_to_geoms(mjcf, geom_material_map)


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
    ET.SubElement(
        visual,
        "global",
        offwidth="1920",
        offheight="1088",
        azimuth="120",
        elevation="-20",
    )

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
        name="isometric_com",
        mode="targetbodycom",
        target="chest",
        pos="2 -2 2",
        xyaxes="1 0 0 0 0 1",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="isometric",
        mode="targetbody",
        target="chest",
        pos="4 -2 1.5",
        zaxis="1 0 0",
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="track_base_link",
        mode="trackcom",
        target="chest",
        pos="2.5 -2.5 1",
        xyaxes="1 1 0 0 1 3",
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


# Example usage:
mjcf = prepend_mujoco_configuration(mjcf)

add_jet_turbine_motors(mjcf)
set_position_actuator_params(mjcf, joint_actuator_params)
add_ft_sites_to_chest(mjcf, robot_urdf, "chest")

# add sites for the imu
add_sites_for_imu(mjcf, robot_urdf)
# add sites for soles
add_sites_to_body(mjcf, robot_urdf, "l_ankle_2", "l_foot_rear")
add_sites_to_body(mjcf, robot_urdf, "r_ankle_2", "r_foot_rear")
add_sites_to_body(mjcf, robot_urdf, "l_ankle_2", "l_foot_front")
add_sites_to_body(mjcf, robot_urdf, "r_ankle_2", "r_foot_front")

# add sensors to the robot
add_sensors_to_sites(mjcf)
# Add chest gyro and velocimeter sensors to the model
sensor = mjcf.find("sensor")
if sensor is None:
    sensor = ET.SubElement(mjcf, "sensor")
ET.SubElement(
    sensor, "gyro", name="chest_gyro_sensor", site="chest_sensor", noise="0.048"
)
ET.SubElement(
    sensor,
    "velocimeter",
    name="chest_velocimeter_sensor",
    site="chest_sensor",
    noise="0.5",
)
# add camera to the robot
for body in mjcf.findall(".//body"):
    if "realsense" in body.attrib["name"]:
        add_camera(body, name=body.attrib["name"], r_y=-np.pi / 2, r_z=np.pi / 2)

# create collision groups and affinities
separate_left_right_collision_groups(mjcf)

# print the model
mjmodel_str = ET.tostring(mjcf, encoding="unicode", method="xml")
# print(mjmodel_str)


# Save the updated MJCF model to the XML file
# Update <meshdir> subelement to be relative to the mujoco_path
# Update <compiler> subelement to set meshdir exactly to "../../meshes/obj"
# Remove duplicate <compiler> elements and set meshdir
compiler_elems = mjcf.findall("compiler")
if compiler_elems:
    # Keep only the first <compiler> element
    first_compiler = compiler_elems[0]
    first_compiler.set("meshdir", mesh_relative_path)
    for compiler_elem in compiler_elems[1:]:
        mjcf.remove(compiler_elem)
else:
    # If no <compiler> exists, add one
    ET.SubElement(mjcf, "compiler", meshdir=mesh_relative_path)

formatted_xml = format_xml(mjcf)
if robot_relative_path.endswith("model_stl.urdf"):
    output_xml_path = os.path.join(mujoco_path, "iRonCub_stl.xml")
else:
    output_xml_path = os.path.join(mujoco_path, "iRonCub.xml")
os.makedirs(mujoco_path, exist_ok=True)
with open(output_xml_path, "w") as f:
    f.write(formatted_xml)

print("MuJoCo configuration prepended and saved to iRonCub.xml")

# Save the model to a temporary file
path_temp_xml = tempfile.NamedTemporaryFile(mode="w+", delete=False)
with open(path_temp_xml.name, "w") as f:
    f.write(formatted_xml)


model = mujoco.MjModel.from_xml_path(output_xml_path)
data = mujoco.MjData(model)

# Visualize the model
mujoco.viewer.launch(model=model, data=data)

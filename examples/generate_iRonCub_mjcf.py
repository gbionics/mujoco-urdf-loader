import tempfile
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import os
import subprocess
import idyntree.swig as idyntree
from mujoco_urdf_loader.generator import load_urdf_into_mjcf

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

### make sure you run this in your terminal before you launch the python script otherwise this doesnot work
#  call <path\to\ironcub_ws\>\install\local_setup.bat
# Print the value of the environment variable
# package = os.getenv("IRONCUB_COMPONENT_SOURCE_DIR")
package = "C:/Users/pvanteddu/Documents/iRonCub_ws/src/component_ironcub"
# Load the robot urdf
robot_relative_path = "models/iRonCub-Mk3/iRonCub/robots/iRonCub-Mk3/model.urdf"
robot_path = os.path.join(package, robot_relative_path)

robot_urdf = get_robot_urdf(robot_path)
# find the mesh path
mesh_relative_path = "models/iRonCub-Mk3/iRonCub/meshes/obj"

mesh_path = os.path.join(package, mesh_relative_path)

# remove the gazebo elements
# robot_urdf = remove_gazebo_elements(robot_urdf)

# add the mujoco element
robot_urdf = add_mujoco_element(robot_urdf, mesh_path)

# Load the converted model to add the actuators, sensors and constraints
mjcf = load_urdf_into_mjcf(robot_urdf)

# create a new worldbody, add the robot to it and remove the old worldbody
add_new_worldbody(mjcf, freeze_root=False)

for joint in mjcf.findall(".//body/joint"):
    # if not any(
    #     joint_element in joint.attrib["name"] for joint_element in hand_elements
    # ):
    ctrlrange = joint.attrib["range"]
    add_position_actuator(
        mjcf,
        joint=joint.attrib["name"],
        ctrlrange=[float(ctrlrange.split()[0]), float(ctrlrange.split()[1])],
        kp=1000,
        name=joint.attrib["name"] + "_motor",
    )

# set the damping
set_joint_damping(mjcf, damping=2)


# set_joint_damping(mjcf, subset=hand_elements, damping=0.005)
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


add_sites_turbines(mjcf, "chest", "sim_sea_l_jet_turbine", "l_jet_turbine")
add_sites_turbines(mjcf, "chest", "sim_sea_r_jet_turbine", "r_jet_turbine")
add_sites_turbines(mjcf, "l_elbow_1", "sim_sea_l_arm_p250", "l_arm_turbine")
add_sites_turbines(mjcf, "r_elbow_1", "sim_sea_r_arm_p250", "r_arm_turbine")


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


add_jet_turbine_motors(mjcf)
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
# add camera to the robot
for body in mjcf.findall(".//body"):
    if "realsense" in body.attrib["name"]:
        add_camera(body, name=body.attrib["name"], r_y=-np.pi / 2, r_z=np.pi / 2)

# create collision groups and affinities
separate_left_right_collision_groups(mjcf)

# print the model
mjmodel_str = ET.tostring(mjcf, encoding="unicode", method="xml")
# print(mjmodel_str)

# Save the formatted model
with open("iRonCub.xml", "w") as f:
    formatted_xml = format_xml(mjcf)
    f.write(formatted_xml)

# Save the model to a temporary file
path_temp_xml = tempfile.NamedTemporaryFile(mode="w+", delete=False)
with open(path_temp_xml.name, "w") as f:
    f.write(formatted_xml)

# Include the model in a simple world
world_str = f"""
<mujoco model="iRonCubCubWorld">
    <include file="{path_temp_xml.name}"/>

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
        <geom name="floor" pos="0 0 -0.78" size="0 0 0.05" type="plane" material="groundplane"/>
    </worldbody>
</mujoco>
"""

# Load the model in mujoco and visualize it
model = mujoco.MjModel.from_xml_string(world_str)
data = mujoco.MjData(model)

# Visualize the model
mujoco.viewer.launch(model=model, data=data)

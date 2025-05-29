import tempfile
import xml.etree.ElementTree as ET

import mujoco


def load_urdf_into_mjcf(robot_urdf: ET.Element) -> ET.Element:
    model_str = ET.tostring(robot_urdf, encoding="unicode", method="xml")

    model = mujoco.MjModel.from_xml_string(model_str)

    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False
    ) as f:  # added delete = false to make it work in windows
        mujoco.mj_saveLastXML(f.name, model)
        mjcf_file = ET.parse(f.name).getroot()

    return mjcf_file

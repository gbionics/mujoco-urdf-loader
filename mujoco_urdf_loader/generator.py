import os
import tempfile
import xml.etree.ElementTree as ET

import mujoco


def load_urdf_into_mjcf(robot_urdf: ET.Element) -> ET.Element:
    model_str = ET.tostring(robot_urdf, encoding="unicode", method="xml")

    model = mujoco.MjModel.from_xml_string(model_str)

    # Use delete=False to allow mujoco to write to the file on Windows
    # (Windows doesn't allow opening a file that's already open)
    f = tempfile.NamedTemporaryFile(mode="w+", suffix=".xml", delete=False)
    try:
        f.close()  # Close the file so mujoco can write to it
        mujoco.mj_saveLastXML(f.name, model)
        mjcf_file = ET.parse(f.name).getroot()
    finally:
        os.unlink(f.name)  # Clean up the temp file

    return mjcf_file

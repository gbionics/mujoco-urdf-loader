import xml.etree.ElementTree as ET

import pytest

from mujoco_urdf_loader import generator


def test_load_urdf_into_mjcf():
    robot_urdf_str = """
    <robot name="ergoCub">
    </robot>
    """
    robot_urdf = ET.fromstring(robot_urdf_str)

    mjcf = generator.load_urdf_into_mjcf(robot_urdf)

    assert mjcf is not None

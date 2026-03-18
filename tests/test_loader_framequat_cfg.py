import xml.etree.ElementTree as ET

import pytest

from mujoco_urdf_loader.loader import FrameQuatSensorCfg, URDFtoMuJoCoLoader, URDFtoMuJoCoLoaderCfg


def _make_empty_mjcf() -> ET.Element:
    return ET.fromstring(
        """
        <mujoco model="test_model">
            <worldbody>
                <body name="base"/>
            </worldbody>
        </mujoco>
        """
    )


def test_add_framequat_sensors_none_keeps_model_unchanged():
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[]))

    loader.add_framequat_sensors(None)

    assert loader.mjcf.find(".//sensor/framequat") is None


def test_add_framequat_sensors_accepts_list_of_dataclasses():
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[]))

    loader.add_framequat_sensors(
        [
            FrameQuatSensorCfg(objname="imu_frame", objtype="site", name="imu_quat"),
            FrameQuatSensorCfg(objname="base", objtype="body", name="base_quat"),
        ]
    )

    imu_sensor = loader.mjcf.find(".//sensor/framequat[@name='imu_quat']")
    base_sensor = loader.mjcf.find(".//sensor/framequat[@name='base_quat']")

    assert imu_sensor is not None
    assert imu_sensor.attrib["objtype"] == "site"
    assert imu_sensor.attrib["objname"] == "imu_frame"

    assert base_sensor is not None
    assert base_sensor.attrib["objtype"] == "body"
    assert base_sensor.attrib["objname"] == "base"


def test_add_framequat_sensors_accepts_dict_with_obtype_alias():
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[]))

    loader.add_framequat_sensors(
        [
            {
                "objname": "imu_frame",
                "obtype": "site",
                "name": "imu_quat",
            }
        ]
    )

    sensor = loader.mjcf.find(".//sensor/framequat[@name='imu_quat']")
    assert sensor is not None
    assert sensor.attrib["objtype"] == "site"
    assert sensor.attrib["objname"] == "imu_frame"


def test_add_framequat_sensors_raises_on_missing_required_fields():
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[]))

    with pytest.raises(ValueError):
        loader.add_framequat_sensors([{"objname": "imu_frame", "name": "imu_quat"}])

import xml.etree.ElementTree as ET

import pytest
import resolve_robotics_uri_py as rru

from mujoco_urdf_loader.loader import FrameQuatSensorCfg, URDFtoMuJoCoLoader, URDFtoMuJoCoLoaderCfg
from mujoco_urdf_loader.urdf_fcn import get_mesh_path


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


def test_add_framequat_sensors_to_ergocub_sn001():
    urdf_root = str(rru.resolve_robotics_uri("package://ergoCub/robots/ergoCubSN001/model.urdf"))
    controlled_joints = [
        "l_hip_pitch",
        "r_hip_pitch",
        "torso_roll",
        "l_hip_roll",
        "r_hip_roll",
        "torso_pitch",
        "torso_yaw",
        "l_hip_yaw",
        "r_hip_yaw",
        "l_shoulder_pitch",
        "neck_pitch",
        "r_shoulder_pitch",
        "l_knee",
        "r_knee",
        "l_shoulder_roll",
        "neck_roll",
        "r_shoulder_roll",
        "l_ankle_pitch",
        "r_ankle_pitch",
        "neck_yaw",
        "l_ankle_roll",
        "r_ankle_roll",
        "l_shoulder_yaw",
        "r_shoulder_yaw",
        "l_elbow",
        "r_elbow",
    ]
    mesh_path = get_mesh_path(ET.parse(urdf_root).getroot())
    framequat_sensors_cfg = [
        FrameQuatSensorCfg(objname="realsense_depth_frame", objtype="site", name="realsense_depth_quat"),
        FrameQuatSensorCfg(objname="realsense_rgb_frame", objtype="site", name="realsense_rgb_quat"),
    ]
    cfg = URDFtoMuJoCoLoaderCfg(
        controlled_joints=controlled_joints,
        framequat_sensors_cfg=framequat_sensors_cfg,
        all_missing_joints_as_sites=True,
    )
    loader = URDFtoMuJoCoLoader.load_urdf(urdf_root, mesh_path, cfg)

    depth_sensor = loader.mjcf.find(".//sensor/framequat[@name='realsense_depth_quat']")
    rgb_sensor = loader.mjcf.find(".//sensor/framequat[@name='realsense_rgb_quat']")

    assert depth_sensor is not None
    assert depth_sensor.attrib["objtype"] == "site"
    assert depth_sensor.attrib["objname"] == "realsense_depth_frame"

    assert rgb_sensor is not None
    assert rgb_sensor.attrib["objtype"] == "site"
    assert rgb_sensor.attrib["objname"] == "realsense_rgb_frame"

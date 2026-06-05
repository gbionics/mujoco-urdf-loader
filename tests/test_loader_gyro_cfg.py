import xml.etree.ElementTree as ET

import resolve_robotics_uri_py as rru

from mujoco_urdf_loader.loader import GyroSensorCfg, URDFtoMuJoCoLoader, URDFtoMuJoCoLoaderCfg
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


def test_add_gyro_sensors_none_keeps_model_unchanged():
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(observed_joints=[]))

    loader.add_gyro_sensors(None)

    assert loader.mjcf.find(".//sensor/gyro") is None


def test_add_gyro_sensors_accepts_list_of_dataclasses():
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(observed_joints=[]))

    loader.add_gyro_sensors(
        [
            GyroSensorCfg(site="imu_frame", name="imu_gyro"),
            GyroSensorCfg(site="torso_imu", name="torso_gyro"),
        ]
    )

    imu_sensor = loader.mjcf.find(".//sensor/gyro[@name='imu_gyro']")
    torso_sensor = loader.mjcf.find(".//sensor/gyro[@name='torso_gyro']")

    assert imu_sensor is not None
    assert imu_sensor.attrib["site"] == "imu_frame"

    assert torso_sensor is not None
    assert torso_sensor.attrib["site"] == "torso_imu"


def test_add_gyro_sensors_accepts_dict():
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(observed_joints=[]))

    loader.add_gyro_sensors(
        [
            {
                "site": "imu_frame",
                "name": "imu_gyro",
            }
        ]
    )

    sensor = loader.mjcf.find(".//sensor/gyro[@name='imu_gyro']")
    assert sensor is not None
    assert sensor.attrib["site"] == "imu_frame"


def test_add_gyro_sensors_accepts_dict_with_objname_alias():
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(observed_joints=[]))

    loader.add_gyro_sensors(
        [
            {
                "objname": "imu_frame",
                "name": "imu_gyro",
            }
        ]
    )

    sensor = loader.mjcf.find(".//sensor/gyro[@name='imu_gyro']")
    assert sensor is not None
    assert sensor.attrib["site"] == "imu_frame"


def test_add_gyro_sensors_to_ergocub_sn001():
    urdf_root = str(rru.resolve_robotics_uri("package://ergoCub/robots/ergoCubSN001/model.urdf"))
    observed_joints = [
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
    gyro_sensors_cfg = [
        GyroSensorCfg(site="realsense_depth_frame", name="realsense_depth_gyro"),
        GyroSensorCfg(site="realsense_rgb_frame", name="realsense_rgb_gyro"),
    ]
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=observed_joints,
        gyro_sensors_cfg=gyro_sensors_cfg,
        all_missing_joints_as_sites=True,
    )
    loader = URDFtoMuJoCoLoader.load_urdf(urdf_root, mesh_path, cfg)

    depth_sensor = loader.mjcf.find(".//sensor/gyro[@name='realsense_depth_gyro']")
    rgb_sensor = loader.mjcf.find(".//sensor/gyro[@name='realsense_rgb_gyro']")

    assert depth_sensor is not None
    assert depth_sensor.attrib["site"] == "realsense_depth_frame"

    assert rgb_sensor is not None
    assert rgb_sensor.attrib["site"] == "realsense_rgb_frame"

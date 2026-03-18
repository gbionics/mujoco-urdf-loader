import xml.etree.ElementTree as ET

from mujoco_urdf_loader.loader import GyroSensorCfg, URDFtoMuJoCoLoader, URDFtoMuJoCoLoaderCfg


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
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[]))

    loader.add_gyro_sensors(None)

    assert loader.mjcf.find(".//sensor/gyro") is None


def test_add_gyro_sensors_accepts_list_of_dataclasses():
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[]))

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
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[]))

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
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[]))

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

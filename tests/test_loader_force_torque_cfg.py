import xml.etree.ElementTree as ET

import pytest
import resolve_robotics_uri_py as rru
import mujoco

from mujoco_urdf_loader.loader import (
    ForceTorqueSensorCfg,
    URDFtoMuJoCoLoader,
    URDFtoMuJoCoLoaderCfg,
)
from mujoco_urdf_loader.urdf_fcn import get_mesh_path


def _make_empty_mjcf() -> ET.Element:
    return ET.fromstring(
        """
        <mujoco model="test_model">
            <worldbody>
                <body name="base">
                    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                    <joint name="test_joint" type="hinge"/>
                </body>
            </worldbody>
            <sensor/>
        </mujoco>
        """
    )


def test_add_force_torque_sensors_none_keeps_model_unchanged():
    loader = URDFtoMuJoCoLoader(
        _make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(observed_joints=[])
    )

    loader.add_force_torque_sensors(None)

    assert loader.mjcf.find(".//sensor/force") is None


def test_add_force_torque_sensors_accepts_list_of_dataclasses():
    loader = URDFtoMuJoCoLoader(
        _make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(observed_joints=[])
    )

    loader.add_force_torque_sensors(
        [
            ForceTorqueSensorCfg(joint="test_joint", sensor_name="test_ft"),
            ForceTorqueSensorCfg(joint="test_joint", sensor_name="test_ft_2", noise=0.005),
        ]
    )

    test_sensor = loader.mjcf.find(".//sensor/force[@name='test_ft']")
    test_sensor_2 = loader.mjcf.find(".//sensor/force[@name='test_ft_2']")

    assert test_sensor is not None
    assert test_sensor.attrib["site"] == "test_joint_ft_site"

    assert test_sensor_2 is not None
    assert test_sensor_2.attrib["site"] == "test_joint_ft_site"
    assert test_sensor_2.attrib["noise"] == "0.005"

    xml_str = loader.get_mjcf_string()
    model = mujoco.MjModel.from_xml_string(xml_str)
    assert model is not None


def test_add_force_torque_sensors_accepts_dict():
    loader = URDFtoMuJoCoLoader(
        _make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(observed_joints=[])
    )

    loader.add_force_torque_sensors(
        [
            {
                "joint": "test_joint",
                "sensor_name": "test_ft",
            }
        ]
    )

    sensor = loader.mjcf.find(".//sensor/force[@name='test_ft']")
    assert sensor is not None
    assert sensor.attrib["site"] == "test_joint_ft_site"


def test_add_force_torque_sensors_accepts_dict_with_optional_fields():
    loader = URDFtoMuJoCoLoader(
        _make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(observed_joints=[])
    )

    loader.add_force_torque_sensors(
        [
            {
                "joint": "test_joint",
                "sensor_name": "test_ft",
                "noise": 0.01,
                "cutoff_frequency": 50,
            }
        ]
    )

    sensor = loader.mjcf.find(".//sensor/force[@name='test_ft']")
    assert sensor is not None
    assert sensor.attrib["site"] == "test_joint_ft_site"
    assert sensor.attrib["noise"] == "0.01"
    assert sensor.attrib["cutoff"] == "50"

    xml_str = loader.get_mjcf_string()
    model = mujoco.MjModel.from_xml_string(xml_str)
    assert model is not None


def test_add_force_torque_sensors_raises_on_missing_required_fields():
    loader = URDFtoMuJoCoLoader(
        _make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(observed_joints=[])
    )

    with pytest.raises(ValueError):
        loader.add_force_torque_sensors([{"sensor_name": "test_ft"}])


def test_add_force_torque_sensors_to_ergocub_sn001():
    urdf_root = str(
        rru.resolve_robotics_uri("package://ergoCub/robots/ergoCubSN001/model.urdf")
    )
    observed_joints = [
        "l_hip_pitch",
        "r_hip_pitch",
        "torso_roll",
    ]
    mesh_path = get_mesh_path(ET.parse(urdf_root).getroot())
    force_torque_sensors_cfg = [
        ForceTorqueSensorCfg(joint="l_hip_pitch", sensor_name="l_hip_pitch_ft"),
        ForceTorqueSensorCfg(
            joint="r_hip_pitch",
            sensor_name="r_hip_pitch_ft",
        ),
    ]
    cfg = URDFtoMuJoCoLoaderCfg(
        observed_joints=observed_joints,
        force_torque_sensors_cfg=force_torque_sensors_cfg,
        all_missing_joints_as_sites=True,
    )
    loader = URDFtoMuJoCoLoader.load_urdf(urdf_root, mesh_path, cfg)

    l_sensor = loader.mjcf.find(".//sensor/force[@name='l_hip_pitch_ft']")
    r_sensor = loader.mjcf.find(".//sensor/force[@name='r_hip_pitch_ft']")

    assert l_sensor is not None
    assert l_sensor.attrib["site"] == "l_hip_pitch_ft_site"

    assert r_sensor is not None
    assert r_sensor.attrib["site"] == "r_hip_pitch_ft_site"

    # Verify MuJoCo can parse the compiled model
    xml_str = loader.get_mjcf_string()
    model = mujoco.MjModel.from_xml_string(xml_str)
    assert model is not None

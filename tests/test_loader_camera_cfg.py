import xml.etree.ElementTree as ET

import pytest

from mujoco_urdf_loader.loader import CameraCfg, URDFtoMuJoCoLoader, URDFtoMuJoCoLoaderCfg


def _make_empty_mjcf() -> ET.Element:
    return ET.fromstring(
        """
        <mujoco model="test_model">
            <worldbody>
                <body name="base">
                    <site name="camera_site" pos="0 0 0" quat="1 0 0 0"/>
                    <site name="camera_site_2" pos="0.1 0.2 0.3" quat="1 0 0 0"/>
                </body>
            </worldbody>
        </mujoco>
        """
    )


def test_add_cameras_none_keeps_model_unchanged():
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[]))

    loader.add_cameras(None)

    assert loader.mjcf.find(".//camera") is None


def test_add_cameras_accepts_list_of_dataclasses():
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[]))

    loader.add_cameras(
        [
            CameraCfg(name="root_depth_camera", site="camera_site", fovy=60.0),
            CameraCfg(name="root_rgb_camera", site="camera_site_2", fovy=75.0),
        ]
    )

    depth_camera = loader.mjcf.find(".//camera[@name='root_depth_camera']")
    rgb_camera = loader.mjcf.find(".//camera[@name='root_rgb_camera']")

    assert depth_camera is not None
    assert depth_camera.attrib["fovy"] == "60.0"

    assert rgb_camera is not None
    assert rgb_camera.attrib["fovy"] == "75.0"


def test_add_cameras_accepts_dict_with_site_key():
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[]))

    loader.add_cameras(
        [
            {
                "name": "root_depth_camera",
                "site": "camera_site",
                "fovy": 60.0,
            }
        ]
    )

    camera = loader.mjcf.find(".//camera[@name='root_depth_camera']")
    assert camera is not None
    assert camera.attrib["fovy"] == "60.0"


def test_add_cameras_accepts_dict_with_link_alias():
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[]))

    loader.add_cameras(
        [
            {
                "name": "root_depth_camera",
                "link": "camera_site",
                "fovy": 60.0,
            }
        ]
    )

    camera = loader.mjcf.find(".//camera[@name='root_depth_camera']")
    assert camera is not None
    assert camera.attrib["fovy"] == "60.0"


def test_add_cameras_raises_on_missing_required_fields():
    loader = URDFtoMuJoCoLoader(_make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[]))

    with pytest.raises(ValueError):
        loader.add_cameras([{"name": "root_depth_camera", "fovy": 60.0}])

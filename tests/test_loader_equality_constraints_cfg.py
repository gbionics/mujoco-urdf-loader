import xml.etree.ElementTree as ET

import pytest

from mujoco_urdf_loader.loader import (
    EqualityConstraintCfg,
    URDFtoMuJoCoLoader,
    URDFtoMuJoCoLoaderCfg,
)


def _make_empty_mjcf() -> ET.Element:
    return ET.fromstring(
        """
        <mujoco model="test_model">
            <worldbody>
                <body name="body_a">
                    <site name="site_a" pos="0 0 0" quat="1 0 0 0"/>
                </body>
                <body name="body_b">
                    <site name="site_b" pos="0.1 0.2 0.3" quat="1 0 0 0"/>
                </body>
            </worldbody>
        </mujoco>
        """
    )


def test_add_equality_constraints_none_keeps_model_unchanged():
    loader = URDFtoMuJoCoLoader(
        _make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[])
    )

    loader.add_equality_constraints(None)

    assert loader.mjcf.find(".//equality") is None


def test_add_equality_constraints_accepts_list_of_dataclasses():
    loader = URDFtoMuJoCoLoader(
        _make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[])
    )

    loader.add_equality_constraints(
        [EqualityConstraintCfg(site1="site_a", site2="site_b")]
    )

    connect = loader.mjcf.find(".//equality/connect")
    assert connect is not None
    assert connect.attrib["site1"] == "site_a"
    assert connect.attrib["site2"] == "site_b"


def test_add_equality_constraints_accepts_dict():
    loader = URDFtoMuJoCoLoader(
        _make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[])
    )

    loader.add_equality_constraints(
        [{"site1": "site_a", "site2": "site_b"}]
    )

    connect = loader.mjcf.find(".//equality/connect")
    assert connect is not None
    assert connect.attrib["site1"] == "site_a"
    assert connect.attrib["site2"] == "site_b"


def test_add_equality_constraints_multiple():
    loader = URDFtoMuJoCoLoader(
        _make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[])
    )

    loader.add_equality_constraints(
        [
            EqualityConstraintCfg(site1="site_a", site2="site_b"),
            EqualityConstraintCfg(site1="site_b", site2="site_a"),
        ]
    )

    connects = loader.mjcf.findall(".//equality/connect")
    assert len(connects) == 2
    assert connects[0].attrib["site1"] == "site_a"
    assert connects[1].attrib["site1"] == "site_b"


def test_add_equality_constraints_weld_type():
    loader = URDFtoMuJoCoLoader(
        _make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[])
    )

    loader.add_equality_constraints(
        [EqualityConstraintCfg(site1="site_a", site2="site_b", constraint_type="weld")]
    )

    weld = loader.mjcf.find(".//equality/weld")
    assert weld is not None
    assert weld.attrib["body1"] == "body_a"
    assert weld.attrib["body2"] == "body_b"


def test_add_equality_constraints_raises_on_missing_fields():
    loader = URDFtoMuJoCoLoader(
        _make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[])
    )

    with pytest.raises(ValueError):
        loader.add_equality_constraints([{"site1": "site_a"}])


def test_add_equality_constraints_raises_on_invalid_type():
    loader = URDFtoMuJoCoLoader(
        _make_empty_mjcf(), URDFtoMuJoCoLoaderCfg(controlled_joints=[])
    )

    with pytest.raises(TypeError):
        loader.add_equality_constraints(["not_a_valid_config"])

import xml.etree.ElementTree as ET

from mujoco_urdf_loader.loader import URDFtoMuJoCoLoader, URDFtoMuJoCoLoaderCfg


def _make_urdf_root() -> ET.Element:
    return ET.fromstring(
        """
        <robot name="test_robot">
            <link name="base"/>
            <link name="l1"/>
            <link name="l2"/>
            <joint name="j1" type="fixed">
                <parent link="base"/>
                <child link="l1"/>
                <origin xyz="1 0 0" rpy="0 0 0"/>
            </joint>
            <joint name="j2" type="fixed">
                <parent link="l1"/>
                <child link="l2"/>
                <origin xyz="0 2 0" rpy="0 0 0"/>
            </joint>
        </robot>
        """
    )


def _make_mjcf_with_base_only() -> ET.Element:
    return ET.fromstring(
        """
        <mujoco model="test_model">
            <worldbody>
                <body name="base"/>
            </worldbody>
        </mujoco>
        """
    )


def test_get_missing_joint_sites_all_flag_collects_missing_urdf_joints():
    urdf_root = _make_urdf_root()
    mjcf = ET.fromstring(
        """
        <mujoco model="test_model">
            <worldbody>
                <body name="base">
                    <joint name="j1" type="hinge"/>
                </body>
            </worldbody>
        </mujoco>
        """
    )

    sites = URDFtoMuJoCoLoader.get_missing_joint_sites(
        urdf_root,
        mjcf,
        all_missing_joints_as_sites=True,
    )

    site_names = {site["name"] for site in sites}
    assert site_names == {"j2"}


def test_add_sites_for_missing_joints_handles_nested_lumped_missing_joints():
    urdf_root = _make_urdf_root()
    fixed_joint_sites = URDFtoMuJoCoLoader.get_missing_joint_sites(
        urdf_root,
        _make_mjcf_with_base_only(),
        all_missing_joints_as_sites=True,
    )

    fixed_joint_sites = [site for site in fixed_joint_sites if site["name"] == "j2"]

    cfg = URDFtoMuJoCoLoaderCfg(controlled_joints=[])
    loader = URDFtoMuJoCoLoader(_make_mjcf_with_base_only(), cfg)

    loader.add_sites_for_missing_joints(fixed_joint_sites)

    site = loader.mjcf.find(".//site[@name='j2']")
    assert site is not None

    site_parent_body = next(
        (
            body
            for body in loader.mjcf.findall(".//body")
            if body.find("./site[@name='j2']") is not None
        ),
        None,
    )
    assert site_parent_body is not None
    assert site_parent_body.attrib["name"] == "base"

    pos = list(map(float, site.attrib["pos"].split()))
    assert pos == [1.0, 2.0, 0.0]

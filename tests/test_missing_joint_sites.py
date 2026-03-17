import xml.etree.ElementTree as ET

from mujoco_urdf_loader.loader import ControlMode, URDFtoMuJoCoLoader, URDFtoMuJoCoLoaderCfg
from mujoco_urdf_loader.urdf_fcn import get_mesh_path

import resolve_robotics_uri_py as rru

import mujoco
import numpy as np
import idyntree.bindings as idyntree

def _make_urdf_root() -> ET.Element:
    return ET.fromstring(
        """
        <robot name="test_robot">
            <link name="base"/>
            <link name="l1"/>
            <link name="l2"/>
            <joint name="j1" type="revolute">
                <parent link="base"/>
                <child link="l1"/>
                <origin xyz="1 0 0" rpy="0 0 0"/>
                <axis xyz="0 0 1"/>
            </joint>
            <joint name="j2" type="revolute">
                <parent link="l1"/>
                <child link="l2"/>
                <origin xyz="0 2 0" rpy="0 0 0"/>
                <axis xyz="0 0 1"/>
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


def test_get_missing_joint_sites_all_flag_collects_missing_urdf_links():
    urdf_root = _make_urdf_root()
    mjcf = ET.fromstring(
        """
        <mujoco model="test_model">
            <worldbody>
                <body name="base">
                    <body name="l1">
                        <joint name="j1" type="hinge"/>
                    </body>
                </body>
            </worldbody>
        </mujoco>
        """
    )

    # j1 is controlled so l1 survives; j2 is NOT controlled so l2 is missing
    sites = URDFtoMuJoCoLoader.get_missing_joint_sites(
        urdf_root,
        mjcf,
        controlled_joints=["j1"],
        all_missing_joints_as_sites=True,
    )

    site_names = {site["name"] for site in sites}
    assert site_names == {"l2"}


def test_add_sites_for_missing_joints_handles_nested_lumped_missing_joints():
    urdf_root = _make_urdf_root()
    # Neither j1 nor j2 are controlled, so both l1 and l2 are missing
    fixed_joint_sites = URDFtoMuJoCoLoader.get_missing_joint_sites(
        urdf_root,
        _make_mjcf_with_base_only(),
        controlled_joints=[],
        all_missing_joints_as_sites=True,
    )

    fixed_joint_sites = [site for site in fixed_joint_sites if site["name"] == "l2"]

    cfg = URDFtoMuJoCoLoaderCfg(controlled_joints=[])
    loader = URDFtoMuJoCoLoader(_make_mjcf_with_base_only(), cfg)

    loader.add_sites_for_missing_joints(fixed_joint_sites)

    site = loader.mjcf.find(".//site[@name='l2']")
    assert site is not None

    site_parent_body = next(
        (
            body
            for body in loader.mjcf.findall(".//body")
            if body.find("./site[@name='l2']") is not None
        ),
        None,
    )
    assert site_parent_body is not None
    assert site_parent_body.attrib["name"] == "base"

    pos = list(map(float, site.attrib["pos"].split()))
    assert pos == [1.0, 2.0, 0.0]

def test_ergocub_sn001_missing_joint_sites():
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
    cfg = URDFtoMuJoCoLoaderCfg(controlled_joints, all_missing_joints_as_sites=True)
    loader = URDFtoMuJoCoLoader.load_urdf(urdf_root, mesh_path, cfg)

    xml_path = "model.xml"
    with open(xml_path, "w") as f:
        f.write(loader.get_mjcf_string(pretty=True))
    print(f"MuJoCo XML model saved to: {xml_path}")

    mjcf_string = loader.get_mjcf_string()

    # ---- MuJoCo setup ----
    mj_model = mujoco.MjModel.from_xml_string(mjcf_string)
    mj_data = mujoco.MjData(mj_model)

    # ---- iDynTree reduced-model setup (using controlled joints) ----
    idt_loader = idyntree.ModelLoader()
    assert idt_loader.loadReducedModelFromFile(urdf_root, controlled_joints)
    reduced_model = idt_loader.model()

    kin_dyn = idyntree.KinDynComputations()
    assert kin_dyn.loadRobotModel(reduced_model)

    n_reduced_dofs = reduced_model.getNrOfDOFs()

    # Map each controlled joint name to its DOF index in the reduced model
    controlled_dof_indices = {}
    for jname in controlled_joints:
        jidx = reduced_model.getJointIndex(jname)
        assert jidx >= 0, f"Joint {jname} not found in the reduced model"
        controlled_dof_indices[jname] = reduced_model.getJoint(jidx).getDOFsOffset()

    # Collect MuJoCo site names that also exist as frames in iDynTree
    all_site_names = [mj_model.site(i).name for i in range(mj_model.nsite)]
    sites_to_check = [
        name for name in all_site_names
        if reduced_model.getFrameIndex(name) >= 0
    ]
    assert len(sites_to_check) > 0, "No sites with matching iDynTree frames found"

    # Get joint limits from MuJoCo for uniform sampling
    joint_limits = np.zeros((len(controlled_joints), 2))
    for i, jname in enumerate(controlled_joints):
        jid = mj_model.joint(jname).id
        if mj_model.jnt_limited[jid]:
            joint_limits[i] = mj_model.jnt_range[jid]
        else:
            joint_limits[i] = [-1.0, 1.0]

    nr_random_joints_cfg = 5

    sites_with_pos_mismatches = set()  # to track which sites had mismatches across all configurations
    sites_with_rot_mismatches = set()  # to track which sites had mismatches across all configurations
    for joint_cfg_nr in range(nr_random_joints_cfg):
        # Sample random joint positions within limits
        s_ctrl = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1])

        # ---- iDynTree: set reduced-model state ----
        s = idyntree.VectorDynSize(n_reduced_dofs)
        s.zero()
        for i, jname in enumerate(controlled_joints):
            s.setVal(controlled_dof_indices[jname], s_ctrl[i])

        ds = idyntree.VectorDynSize(n_reduced_dofs)
        ds.zero()
        gravity = idyntree.Vector3()
        gravity.zero()
        gravity.setVal(2, -9.81)

        kin_dyn.setRobotState(
            idyntree.Transform.Identity(), s,
            idyntree.Twist(), ds, gravity,
        )

        # ---- MuJoCo: set qpos and run forward kinematics ----
        mj_data.qpos[:] = 0.0
        mj_data.qpos[3] = 1.0  # identity quaternion (w, x, y, z)
        for i, jname in enumerate(controlled_joints):
            jid = mj_model.joint(jname).id
            addr = mj_model.jnt_qposadr[jid]
            mj_data.qpos[addr] = s_ctrl[i]

        mujoco.mj_forward(mj_model, mj_data)

        # ---- Compare FK for every site ----
        site_checked = 0
        for site_name in sites_to_check:
            site_checked += 1
            site_id = mj_model.site(site_name).id
            mj_pos = mj_data.site_xpos[site_id]
            mj_rot = mj_data.site_xmat[site_id].reshape(3, 3)

            tf = kin_dyn.getWorldTransform(site_name)
            idt_pos = np.array([tf.getPosition().getVal(k) for k in range(3)])
            idt_rot = np.array(
                [[tf.getRotation().getVal(r, c) for c in range(3)] for r in range(3)]
            )

            # instead of interrumpting the test at the first failure, we check all sites and report all mismatches at the end

            try:
                np.testing.assert_allclose(
                    mj_pos, idt_pos, atol=1e-3,
                    err_msg=f"Position mismatch for site '{site_name}' in joint configuration #{joint_cfg_nr}",
                )
            except AssertionError as e:
                # print(e) 
                sites_with_pos_mismatches.add(site_name)

            try:
                np.testing.assert_allclose(
                    mj_rot, idt_rot, atol=1e-3,
                    err_msg=f"Rotation mismatch for site '{site_name}' in joint configuration #{joint_cfg_nr}",
                )
            except AssertionError as e:
                # print(e) 
                sites_with_rot_mismatches.add(site_name)

    assert len(sites_with_pos_mismatches) == 0, f"Position mismatches found for sites: {sites_with_pos_mismatches}"
    assert len(sites_with_rot_mismatches) == 0, f"Rotation mismatches found for sites: {sites_with_rot_mismatches}"

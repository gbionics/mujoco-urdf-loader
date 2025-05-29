# mujoco-urdf-loader

This repository contains a collection of scripts for generating MuJoCo (MJCF) files from Unified Robot Description Format (URDF) files. The scripts are designed to work with the [ergoCub](https://ergocub.eu/) robot. 

### Installation

A conda environment with minimal dependencies can be created using the `environment.yaml` file:

```bash
conda env create -f environment.yaml
```

Activate the environment before proceeding:

```bash
conda activate mujocoloaderenv
```

Then, install the package in editable mode:

```bash
pip install -e .
```

**Note:** To run the examples, you need to install [`ergocub-software`](https://github.com/icub-tech-iit/ergocub-software) to allow `resolve_robotics_uri_py` to find the original URDF files. This can be done by installing the `ergocub-software` conda package.

### Examples

This repository includes Python scripts that demonstrate how to use the package to generate MJCF files for the ergoCub robot. The following examples are provided:

* `generate_ergoCub_mjcf.py`: Generates a complete MJCF model of the ErgoCub robot, including a floating base and position actuators for all joints (excluding hands, see note below).
* `generate_ergoCub_torso.py`: Generates an MJCF model of the ErgoCub torso only, excluding the legs. Similar to the full model, it includes position actuators for all joints (excluding hands).
* `generate_ergoCub_hand.py`: Generates an MJCF model of a single ErgoCub hand (right hand). This script specifically handles adding position servos only for actuated joints and uses equality constraints to simulate hand linkages (a technique also used in the other models). Additionally, it allows you to modify the thumb orientation.
* `generate_ergoCub_xml.py`: Generates the MJCF model of a robot through the `URDFtoMuJoCoLoader` class, which takes as input the robot urdf, and controlled joints lists informations in the form of `URDFtoMuJoCoLoaderCfg`. The scripts also loades the model in a wrapper called `MujocoWrapper` that handles simple calls to get robot's pose and joints quantities. 

All examples load the `package://ergoCub/robots/ergoCubSN001/model.urdf` model, convert it to MJCF format, save the resulting file, and then display the model in a simple MuJoCo viewer window.

### Codebase Structure

The codebase is organized into four main Python files:

* `generator.py`: Contains general-purpose functions for generating MJCF files, including the `load_urdf_into_mjcf` function.
* `hands_fcn.py`: Contains functions specifically designed to handle hand models, which often require special considerations like linkages, different actuator gains, and joint damping.
* `mjcf_fcn.py`: Provides functions for working with existing MJCF files, such as adding actuators and sensors.
* `urdf_fcn.py`: Offers functions for manipulating URDF files, including stripping unnecessary parts and finding mesh locations.

### Testing

A small and incomplete test suite is included and can be run using pytest.

# RS-ModCubes: Self-Reconfigurable, Scalable Modular Cubic Robots for Underwater Operations

This repository provides the full system implementation for **RS-ModCubes**, a modular and reconfigurable underwater robotic platform designed for scalable, task-adaptive operations. RS-ModCubes are capable of 6-DoF control, self-docking via onboard electromagnets, and autonomous trajectory tracking and assembly.

 **Paper**: [RS-ModCubes: Self-Reconfigurable, Scalable Modular Cubic Robots for Underwater Operations](https://doi.org/10.1109/LRA.2025.3543139)  
 **Project Page**: [https://jiaxi-zheng.github.io/ModCube.github.io](https://jiaxi-zheng.github.io/ModCube.github.io)

> This codebase is a second-stage development based on the simulation and software stack provided by the **CMU TartanAUV Team**, specifically the "Kingfisher" AUV platform.

---

##  Package Overview

```bash
packages/
├── modcube_common              # Shared control logic, services, and helper scripts
├── modcube_config              # Robot URDFs, parameter files, and sim configs
│   ├── modcube_description     # Mechanical design and static config
│   └── modcube_sim_description # Gazebo-compatible description
├── modcube_mission             # Mission-level planning and teleoperation
├── modcube_msgs                # Custom ROS message and service definitions
├── modcube_sim                 # Main Gazebo simulation launch files
├── modcube_sim_gazebo_plugins  # Custom underwater physics plugins
├── modcube_sim_worlds          # Predefined simulation environments
├── modcube_vehicle             # Vehicle abstraction layers
└── uuv_simulator               # Dependencies for underwater hydrodynamics
Getting Started
1. Launch the Simulation Environment
bash
Copy
Edit
roslaunch modcube_sim kingfisher_umd_sim.launch
This brings up the RS-ModCube simulation in a Gazebo world with underwater dynamics enabled.

2. Start Mission Control & Teleoperation
bash
Copy
Edit
roslaunch modcube_mission teleop_mission.launch
This node handles manual or scripted control of the robot and includes keyboard/joystick-based teleoperation.

3. Send a Navigation Command
bash
Copy
Edit
tap in goto 2 2 2 1 1 1 1
2 2 2 represents the target position (x, y, z)

1 1 1 1 represents the target orientation quaternion (qx, qy, qz, qw)

This command interfaces with the mission planner to reconfigure or move the robot in the desired direction.

Key Features
Self-reconfiguration via magnetic docking with tolerance-guided alignment

Monte Carlo–based hydrodynamic modeling using frontal drag approximations

Model-based PD control with thrust allocation for power efficiency

Minimum snap trajectory generation for smooth, accurate path following

Benchmarking tools for actuation capability and morphological analysis

Example Use Cases
Spiral trajectory tracking in 3D

Two-module autonomous docking in constrained space

Multi-module formation with Möbius trajectory planning

For visuals and demos, see our project site.

Citation
If you use this repository or refer to our methodology, please cite the following publication:

bibtex
Copy
Edit
@article{zheng2025rsmodcubes,
  title     = {RS-ModCubes: Self-Reconfigurable, Scalable Modular Cubic Robots for Underwater Operations},
  author    = {Zheng, Jiaxi and Dai, Guangmin and He, Botao and Mu, Zhaoyang and Meng, Zhaochen and Zhang, Tianyi and Zhi, Weiming and Fan, Dixia},
  journal   = {IEEE Robotics and Automation Letters},
  volume    = {10},
  number    = {4},
  pages     = {3534--3541},
  year      = {2025},
  publisher = {IEEE}
}
🤝 Acknowledgements
This work would not have been possible without the support of my co-authors, mentors, and the collaborative teams at Carnegie Mellon University and Westlake University.
We also gratefully acknowledge the foundational work done by the CMU TartanAUV team.

For questions or contributions, feel free to open an issue or contact me via the project website.

yaml
Copy
Edit

---

Let me know if you'd like to generate READMEs per sub-package (e.g., `modcube_mission/README.md`)
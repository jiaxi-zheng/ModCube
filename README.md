# RS-ModCubes: Self-Reconfigurable, Scalable Modular Cubic Robots for Underwater Operations

This repository contains the simulation, control, and mission planning framework for **RS-ModCubes** a reconfigurable, scalable modular underwater robot system. 

🔗 **Paper**: [RS-ModCubes: Self-Reconfigurable, Scalable Modular Cubic Robots for Underwater Operations](https://doi.org/10.1109/LRA.2025.3543139)  
🌐 **Project Website**: [https://jiaxi-zheng.github.io/ModCube.github.io](https://jiaxi-zheng.github.io/ModCube.github.io)

> _This code has been second-developed based on the original framework from the CMU TartanAUV Team (https://www.tartanauv.com/) (Kingfisher)._


## Repository Structure

```bash
packages/
├── modcube_common              # Shared utilities and core planning/control logic
├── modcube_config              # Configuration files and URDFs
├── modcube_mission             # Mission execution and teleoperation modules
├── modcube_msgs                # Custom ROS messages and service definitions
├── modcube_sim                 # Gazebo simulation interface
├── modcube_sim_gazebo_plugins  # Plugins for Gazebo
├── modcube_sim_worlds          # Simulation environments
├── modcube_vehicle             # Vehicle-specific modules and configurations
└── uuv_simulator               # Underwater vehicle simulator dependencies
```

## Quick Start

Launch the Simulation Environment

```
roslaunch modcube_sim kingfisher_umd_sim.launch
```

Launch Mission Teleoperation

```
roslaunch modcube_mission teleop_mission.launch
```

Set a Navigation Goal

tap in 

```
goto 2 2 2 1 1 1 1
```

2 2 2 → Target position (x, y, z) // 
1 1 1 1 → Target orientation quaternion (qx, qy, qz, qw)
##

Citation
If you use or reference this work, please cite our paper:

```
bibtex
@article{zheng2025rs,
  title={Rs-modcubes: Self-reconfigurable, scalable, modular cubic robots for underwater operations},
  author={Zheng, Jiaxi and Dai, Guangmin and He, Botao and Mu, Zhaoyang and Meng, Zhaochen and Zhang, Tianyi and Zhi, Weiming and Fan, Dixia},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```

License
This project is licensed under the MIT License. See the LICENSE file for details.
---

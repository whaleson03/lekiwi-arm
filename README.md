# LeKiwi Arm: Real · Sim · Frames

**[Live demo →](https://whaleson03.github.io/lekiwi-arm/)**

The SO-101 arm of a LeKiwi robot, shown three ways side by side, all doing the same move from the rest pose to the zero pose:

- **Real**: the robot itself.
- **Sim**: its MuJoCo simulation.
- **Frames**: only the arm's seven coordinate frames, F0 (arm base) to F6 (gripper motor).

Below the videos the simulation runs live in the browser: drag the arm, set the motors in LeRobot units, show the frames and try forward and inverse kinematics.

Robot models adapted from [Ekumen-OS/lekiwi](https://github.com/Ekumen-OS/lekiwi) and [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)'s SO-101 (Apache-2.0; licenses in `web_sim/model/`).

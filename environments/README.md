# environments

This directory stores the different environments that are possible with this project.

For a good starting point, we incorporated the `CartPoleEnv`.
Later on, we utilized the simulations `SimRobotEnv`.
In the end, we tested it on the real environment `RealRobotEnv`.

(Note that, in order to perform the simulation with the animations you have to install `ffmpeg` locally!)

It further holds a wrapper (`wrappers/`) that is used for easier logging to wandb.

In order to create the simulation, the simulated objects and robot is stored in xml files in the directory `models/`.
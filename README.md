Winning Submission of Phase 3 of the 2020 Real Robot Challenge
==============================================================

**This branch contains a version of our Phase 3 submission that solves the tasks with the cube object from Phase 2 of the competition.**

<p align="center">
  <img width="300" height="300" src="https://i.imgur.com/1prGQhx.gif">
</p>

This repository contains code for the winning submission of Phase 3 of the 2020 [Real Robot Challenge](https://real-robot-challenge.com).

A report detailing our approach can be found [here](http://arxiv.org/abs/2101.02842).

This submission is the joint work of
[Charles Schaff](https://ttic.uchicago.edu/~cbschaff/),
[Takuma Yoneda](https://takuma-ynd.github.io/about/),
[Takahiro Maeda](https://github.com/meaten), and
[Matthew R Walter](https://ttic.uchicago.edu/~mwalter/).


This repository is structured as a catkin package and builds on the
[example package](https://github.com/rr-learning/rrc_example_package) provided by the competition,
and [this planning library](https://github.com/yijiangh/pybullet_planning).


## Running the code in simulation

**Make sure to download the Phase 2 image**

To run the code locally, first install [Singularity](https://sylabs.io/guides/3.5/user-guide/quick_start.html)
and download this [singularity image](https://people.tuebingen.mpg.de/felixwidmaier/realrobotchallenge/robot_phase/singularity.html#singularity-download-image)
from the competition. No custom dependencies are required.

Use the `run_locally.sh` script to build the catkin workspace and run commands
inside the singularity image.
For example, to run our controller on a random goal of difficulty 4, use the following command:
```bash
./run_locally.sh /path/to/singularity/image.sif rosrun rrc run_local_episode.py 4
```


## Running the code on the robot cluster

For detailed instructions on how to run this code on the robot cluster, see [this](https://people.tuebingen.mpg.de/felixwidmaier/realrobotchallenge/robot_phase/submission_system.html) page.

This repository contains code for automatically submitting jobs and analyzing logs in the [log_manager](https://github.com/ripl-ttic/rrc_phase_3/tree/cleanup/log_manager) directory.
Note that running jobs on the robot cluster requires an account from the competition organizers.

## Links
- [Videos (phases 1 - 3)](https://youtube.com/playlist?list=PLBUWL2_ywUvE_czrinTTRqqzNu86mYuOV)
- [Report](http://arxiv.org/abs/2101.02842)

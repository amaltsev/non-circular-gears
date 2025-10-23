# [Computational Design and Optimization of Non-Circular Gears (Eurographics 2020)](https://appsrv.cse.cuhk.edu.hk/~haoxu/projects/compute_gear/)
Hao Xu*, Tianwen Fu*, Peng Song, Mingjun Zhou, Chi-Wing Fu, and Niloy J. Mitra (*joint first authors)

## About this project
The following image illustrates this project:
![Our result](./image/teaser.png)
This is an automatic method to design non-circular gears, which takes two shapes as inputs.
The generated gears are optimized not only to resemble the input shapes (left) but also to transfer motion continuously and smoothly (middle). Further, our results can be 3D-printed and put to work in practice (right). 
See implementation details in our [project homepage](https://appsrv.cse.cuhk.edu.hk/~haoxu/projects/compute_gear/).

## Installation
This project is written in python, and is based on the following packages:
- shapely
- NumPy
- matplotlib
- scipy
- openmesh
- pyyaml

## Quick start

- Change into `python_dual_gear` folder
- Run `python main_program.py samples/task_squares.yml`
- View `task_squares.svg`
- Copy and adjust the task, try again

The task is loaded over the defaults in `optimization_config.yaml`
file. For a quick and dirty shapes override, use this:

```
python main_program.py samples/task_generic.yml disney/minnie disney/mickey
```

To run without graphical output, use this (on Linux):
```
QT_QPA_PLATFORM=offscreen python main_program.py samples/task_squares.yml
```

More details and intermediate images are stored in timestamped
subfolders of `debug` folder, with `recent` linking to the most recent
run.

For more details and parameters explanation, refer to the source code.

## Questions
If you met problems, or have questions on this project, don't hesitate to contact us at 
[kevin.futianwen@gmail.com] or [haoxu@cse.cuhk.edu.hk]

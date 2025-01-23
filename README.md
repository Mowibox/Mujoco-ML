# Mujoco-ML
Machine learning project to control a robotic arm in the MuJoCo simulation environment.

# How to run the robot simulations 

Download the required packages:
```
pip install -r requirements.txt
```

Download the repository:

```bash
git clone https://github.com/Mowibox/Mujoco-ML.git
```

Run inside the repository:

    python run.py [options]

        usage: run.py [-h] [-env ENV] [-steps STEPS] [-seed SEED] [--render] [--log]

        optional arguments:
          -h, --help    show this help message and exit
          -env ENV      environment [r2,r3,r5] (default: r2)
          -steps STEPS  Execution steps (default: 10,000)
          -seed SEED    Random seed (default: 1000)
          --render      Enable rendering
          --log         Enable data log


Press 'ESC' in the GUI or CTRL-C in the terminal to exit the simulation.

Commands examples:

    python run.py -env r2 -seed 1000 -steps 100000 --render


Environments description:

    r2: 2D robot with 2 joints
    r3: 2D robot with 3 joints
    r5: 3D robot with 5 joints
    
You can log data by using the following command:

    python run.py -env r2 -seed 1000 -steps 100000 --log > logfile.csv

This will create a file 'logfile.csv' which stores simulation data points for the 2D robot with 2 joints environment, the random seed n°1000 and 100000 steps
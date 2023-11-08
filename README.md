![image](https://github.com/mishraanuraagx/RL_A2C/assets/gym_animation.gif)

# RL_A2C
Actor Critic RL implementation. (gym - Cartpole-v1)


## Environment Setup

- requirements.txt file can be used to setup a conda env with the following command:
- conda create --name <env_name> --file requirements.txt

## Activate your newly created conda env

- conda activate <env_name>

# Running A2C
In the a2c.py file, change value of learn:bool to True to learn and set it to False if you wish to run from a trained model. 
If learn is set to false, make sure the timestamp of the saved model should also be set in variable load_timestamp.

then run the a2c.py implementation with the command:- python a2c.py


## A2C training graph
![image](https://github.com/mishraanuraagx/RL_A2C/assets/24863779/f27b278f-60d8-441e-a833-8c29f2c39490)


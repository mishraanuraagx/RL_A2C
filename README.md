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

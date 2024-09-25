### Drone racing in gym-pybullet-drones using reinforcement learning
- In this project, I use reinforcement learning [PPO](https://github.com/nikhilbarhate99/PPO-PyTorch) to control drone fly through gates in [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones/)
- I refer paper [Reaching the Limit in Autonomous Racing](https://arxiv.org/abs/2310.10943) for the initial setting of reinforcement learning algorithm
#### 25/09/2024 Initial result
* I do initial experiments to figure out that does RL setting work in Gym Pybullet Drones environment (Updating)

##### Using CrazyFly(X type) with simple racing setting

![alt text](https://github.com/phuongboi/drone-racing-using-reinforcement-learning/blob/main/results/racing1.gif)

#### How to use
* Follow author's guide to install gym-pybullet-drones environment
* Training `python train_racing.py`
* Test pretrained model ` python test_racing.py`

### References
* https://github.com/utiasDSL/gym-pybullet-drones/
* https://github.com/nikhilbarhate99/PPO-PyTorch
* Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
* Song, Yunlong, et al. "Reaching the limit in autonomous racing: Optimal control versus reinforcement learning." Science Robotics 8.82 (2023): eadg1462.
# Deep Q Learning - LunarLander
This is the implementation of the deep Q-learning [[paper](https://arxiv.org/abs/1312.5602)] to solve the LunarLander-v2 environment from OpenAI  [[Gym](https://www.gymlibrary.dev/)]. The environment is considered solved when achieving a final score of 200 points or higher on average over the last 100 consecutive runs. 

## Dependencies
Python==3.7.16 

gym==0.17.2

box2d==2.3.10

torch==1.12.1

torchvision==0.13.1

numpy==1.21.5

matplotlib==3.7.1 

tqdm==4.65.0 

## Repository Structure
- `main.py` contains the script to use train an agent using deep reinforcement learning to learn to land a LunarLander. To train the model from scratch, simply run
```python
python main.py
```

## Results
The LunarLander agent is able to reach the target score of 200 by 526 episodes.
![The agent's score of the last 100 training episoides]()

### Contact
This is my re-implementation of the paper. If there are any questions, please contact **Whitney Chiu** <wchiu@gatech.edu>

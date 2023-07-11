import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

class Perceptron(nn.Module):
    def __init__(self, env):
        super(Perceptron, self).__init__()
        torch.manual_seed(0)
        self.fc_layer0 = nn.Linear(env.observation_space.shape[0], 64)
        self.fc_layer1 = nn.Linear(64, 64)
        self.fc_layer2 = nn.Linear(64, env.action_space.n)
        self.relu = torch.nn.ReLU()
        
    def forward(self, state):
        x = self.fc_layer0(state)
        x = self.relu(x)
        x = self.fc_layer1(x)
        x = self.relu(x)
        out = self.fc_layer2(x)
        return out
    
class ExperienceReplay:
    def __init__(self, env, n_memory, n_batch):
        random.seed(0)
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.memory = deque(maxlen = n_memory)  
        self.batch_size = n_batch
        
    def memory_size(self):
        return len(self.memory)
    
    def memorize_exp(self, state, action, next_state, reward, is_terminal):
        e = [state, action, next_state, reward, is_terminal]
        self.memory.append(e)
    
    def get_minibatch_exp(self):
        experiences = random.sample(self.memory, self.batch_size)

        states = experiences[0][0]
        actions = experiences[0][1]
        next_states = experiences[0][2]
        rewards = experiences[0][3]
        is_terminals = experiences[0][4]

        for e in experiences:
            states = torch.from_numpy(np.vstack((e[0], states))).float().to(device)
            actions = torch.from_numpy(np.vstack((e[1], actions))).to(torch.int64).to(device)
            next_states = torch.from_numpy(np.vstack((e[2], next_states))).float().to(device)
            rewards = torch.from_numpy(np.vstack((e[3], rewards))).float().to(device)
            is_terminals = torch.from_numpy(np.vstack((e[4], is_terminals))).float().to(device)

        minibatch_exp = (states, actions, rewards, next_states, is_terminals)
  
        return minibatch_exp

class LunarLanderAgent():
    def __init__(self, env):
        random.seed(0)
        np.random.seed(0)
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        # Initialize q model and target model
        self.q_model = Perceptron(env).to(device)
        self.target_model = Perceptron(env).to(device)
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.q_model.parameters(), learning_rate_)

        # Initialize time step counter and memory
        self.curr_time_step = 0
        self.memory = ExperienceReplay(env, n_memory=memory_size_, n_batch=exp_batch_size_)
        
    def epsilon_greedy(self, state, epsilon=0.):
        # Choose action by e-greedy rule
        state = torch.from_numpy(state).float()
        state = state.unsqueeze(0).to(device)

        if random.random() < epsilon:
            a = np.random.randint(0, self.n_actions)

        else:
            self.q_model.eval()
            with torch.no_grad():
                action_values = self.q_model(state)
            self.q_model.train()

            a = np.argmax(action_values.cpu().data.numpy())
        
        return a
    
    def update_agent(self, state, action, next_state, reward, terminal):
        # Add experience into memory
        self.memory.memorize_exp(state, action, next_state, reward, terminal)
        
        # To stabalize network, only update every n time_steps
        self.curr_time_step += 1

        if self.curr_time_step % update_freq_ == 0:
            if self.memory.memory_size() > exp_batch_size_:
                experiences = self.memory.get_minibatch_exp()
                self.train(experiences, gamma_, tau_)

    def train(self, experiences, gamma, tau):
        self.optimizer.zero_grad()

        states = experiences[0]
        actions = experiences[1]
        rewards = experiences[2]
        next_states = experiences[3]
        terminals = experiences[4]

        # print("======= experiences========")
        # print(experiences)
        # print(dones)
        # print("==== dones =====")
        # print(1-dones)

        # Get predicted q with current state
        # predicted_q = self.q_model(states)
        # print("==== predicted_q =====")
        # print(predicted_q)
        predicted_q = torch.gather(self.q_model(states), 1, actions)

        # predicted_q = self.q_model(states).gather(1, actions)
        # print("==== predicted_q gather=====")
        # print(predicted_q)

        # Get best target q
        best_next_target_q = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        target_q = rewards + gamma * best_next_target_q * (1 - terminals)

        # Calculate loss
        loss = self.loss_function(predicted_q, target_q)
        loss.backward()
        self.optimizer.step()

        # Update target model using soft update
        for target_model_param, q_model_param in zip(self.target_model.parameters(), self.q_model.parameters()):
            target_model_param.data.copy_(tau * q_model_param.data + (1.0 - tau) * target_model_param.data)                   

def DeepQNetwork(n_episodes=2000, timesteps_allowed=1000, init_epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.995, target_score= 200.0):

    all_scores = []                        
    last_100_scores = deque(maxlen = 100)  
    curr_epsilon = init_epsilon                    

    for i in tqdm(range(n_episodes)):
        state = env.reset()
        score = 0
        is_terminal = False

        for j in range(timesteps_allowed):
            while is_terminal == False:
                env.render()
                action = agent.epsilon_greedy(state, curr_epsilon)
                next_state, reward, is_terminal, debug_info = env.step(action)

                agent.update_agent(state, action, next_state, reward, is_terminal)
                state = next_state
                score += reward

        # Append results
        last_100_scores.append(score)
        all_scores.append(score)
        
        # Decay epsilon for next iter
        if curr_epsilon > min_epsilon:
            curr_epsilon *= epsilon_decay

        else:
            curr_epsilon = curr_epsilon

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(last_100_scores)), end="")
        if i % 100 == 0:
            print("\rEpisode {}\tAverage Score: {:.2f}".format(i, np.mean(last_100_scores)))
        if np.mean(last_100_scores) >= target_score:
            print("\n Target score reached in {:d} episodes. \tAverage Score: {:.2f}".format(i-100, np.mean(last_100_scores)))
            torch.save(agent.q_model.state_dict(), "checkpoint.pth")
            break

    return all_scores, last_100_scores

def plot_all_score_curve(folder, scores, lr=5e-4, gamma=0.99, epsilon_decay=0.995):
    plt.close()
    plt.figure()
    plt.plot(np.arange(len(scores)), scores, label = "score")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    
    plt.title(f"Score of LunarLander by Episode, \n (lr={lr}, gamma={gamma}, epsilon decay={epsilon_decay})")
    plt.savefig(f"{folder}/training_score_curve.png")
    plt.show()
            

def plot_reward_of_last_100_episode(folder, last_100_scores, lr=5e-4, gamma=0.99, epsilon_decay=0.995):
    plt.close()
    plt.figure()
    plt.plot(np.arange(len(last_100_scores)), last_100_scores, label = "score")
    plt.xlabel("Last 100 episode")
    plt.ylabel("Score")
    plt.legend()
    
    plt.title(f"Score of LunarLander by Last 100 Episodes, \n (lr={lr}, gamma={gamma}, epsilon decay={epsilon_decay})")
    plt.savefig(f"{folder}/training_last_100_score_curve.png")
    plt.show()


def epsilon_decay_experiment():
    print('Running epsilon decay experiment...')
    global env, agent, memory_size_, exp_batch_size_, gamma_, tau_, epsilon_decay_, update_freq_, target_score_, learning_rate_, device

    env = gym.make('LunarLander-v2')
    env.seed(0)
    
    # Hyperparameters
    memory_size_ = 100000  
    exp_batch_size_ = 64   
    gamma_ = 0.99         
    learning_rate_ = 5e-4        
    tau_ = 1e-3              
    update_freq_ = 4
    target_score_ = 200

    test_epsilon_decay = [0.7, 0.8, 0.9, 0.995]     

    # Initialization
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    for eps in test_epsilon_decay:
        epsilon_decay_ = eps
        print(f"epsilon decay = {epsilon_decay_}") # CHANGE
        agent = LunarLanderAgent(env)
        scores, last_100_scores = DeepQNetwork(target_score = target_score_)
        plot_all_score_curve(f"epsilon_decay_exp_{epsilon_decay_}", scores, epsilon_decay=epsilon_decay_)
        plot_reward_of_last_100_episode(f"epsilon_decay_exp_{epsilon_decay_}", last_100_scores, epsilon_decay=epsilon_decay_)

def gamma_experiment():
    print('Running discount factor experiment...')
    global env, agent, memory_size_, exp_batch_size_, gamma_, tau_, epsilon_decay_, update_freq_, target_score_, learning_rate_, device

    env = gym.make('LunarLander-v2')
    env.seed(0)
    
    # Hyperparameters
    memory_size_ = 100000
    exp_batch_size_ = 64         
    learning_rate_ = 5e-4        
    tau_ = 1e-3
    epsilon_decay_ = 0.995                  
    update_freq_ = 4
    target_score_ = 200

    test_gamma_ = [0.7, 0.8, 0.9, 0.99]     

    # Initialization
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    for g in test_gamma_:
        gamma_ = g
        print(f"discount factor = {gamma_}") # CHANGE
        agent = LunarLanderAgent(env)
        scores, last_100_scores = DeepQNetwork(epsilon_decay = epsilon_decay_, target_score = target_score_)
        plot_all_score_curve(f"gamma_exp_{gamma_}", scores, gamma=gamma_)
        plot_reward_of_last_100_episode(f"gamma_exp_{gamma_}", last_100_scores, gamma=gamma_)

def lr_experiment():
    print('Running learning rate experiment...')
    global env, agent, memory_size_, exp_batch_size_, gamma_, tau_, epsilon_decay_, update_freq_, target_score_, learning_rate_, device

    env = gym.make('LunarLander-v2')
    env.seed(0)
    
    # Hyperparameters
    memory_size_ = 100000
    exp_batch_size_ = 64         
    gamma_ = 0.99            
    tau_ = 1e-3     
    epsilon_decay_ = 0.995         
    update_freq_ = 4
    target_score_ = 200

    test_learning_rates_ = [1e-1, 1e-2, 1e-3, 5e-4]     

    # Initialization
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    for lr in test_learning_rates_:
        learning_rate_ = lr
        print(f"learning rate = {learning_rate_}") # CHANGE
        agent = LunarLanderAgent(env)
        scores, last_100_scores = DeepQNetwork(epsilon_decay = epsilon_decay_, target_score = target_score_)
        plot_all_score_curve(f"lr_exp_{learning_rate_}", scores, lr=learning_rate_)
        plot_reward_of_last_100_episode(f"lr_exp_{learning_rate_}", last_100_scores, lr=learning_rate_)


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    env.seed(0)

    # Hyperparameters
    memory_size_ = 100000
    exp_batch_size_ = 64         
    gamma_ = 0.99            
    tau_ = 1e-3
    epsilon_decay_ = 0.995              
    learning_rate_ = 5e-4     
    update_freq_ = 4
    target_score_ = 200

    # Initialization
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    agent = LunarLanderAgent(env)
    scores, last_100_scores = DeepQNetwork(epsilon_decay = epsilon_decay_, target_score = target_score_)

    # Plot score learning curve
    plot_all_score_curve("figures", scores)
    plot_reward_of_last_100_episode("figures", last_100_scores)

    # Hyperparameter testing #Optional
    # lr_experiment()
    # gamma_experiment()
    # epsilon_decay_experiment()

    

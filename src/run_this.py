from env import Env
from DDPG import DDPG
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import csv

#####################  hyper parameters  ####################
CHECK_EPISODE = 4
LEARNING_MAX_EPISODE = 50
MAX_EP_STEPS = 3000
TEXT_RENDER = False
SCREEN_RENDER = False
CHANGE = False
SLEEP_TIME = 0.1

#####################  function  ####################
def exploration (a, r_dim, b_dim, r_var, b_var):
    for i in range(r_dim + b_dim):
        # resource
        if i < r_dim:
            a[i] = np.clip(np.random.normal(a[i], r_var), 0, 1) * r_bound
        # bandwidth
        elif i < r_dim + b_dim:
            a[i] = np.clip(np.random.normal(a[i], b_var), 0, 1) * b_bound
    return a

###############################  training  ####################################

if __name__ == "__main__":
    start_time = time.time()
    env = Env()
    s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, task_inf, limit, location = env.get_inf()
    ddpg = DDPG(s_dim, r_dim, b_dim, o_dim, r_bound, b_bound)

    r_var = 1  # control exploration
    b_var = 1
    ep_reward = []
    r_v, b_v = [], []
    var_reward = []
    max_rewards = 0
    episode = 0
    var_counter = 0
    epoch_inf = []
    transitions = []
    prev_state = []
    while var_counter < LEARNING_MAX_EPISODE:
        # initialize
        s = env.reset()
        ep_reward.append(0)
        if SCREEN_RENDER:
            env.initial_screen_demo()

        for j in range(MAX_EP_STEPS):
            time.sleep(SLEEP_TIME)
            # render
            if SCREEN_RENDER:
                env.screen_demo()
            if TEXT_RENDER and j % 30 == 0:
                env.text_render()

            # DDPG
            # choose action according to state
            a = ddpg.choose_action(s)  # a = [R B O]
            # add randomness to action selection for exploration
            a = exploration(a, r_dim, b_dim, r_var, b_var)
            # store the transition parameter
            s_, r = env.ddpg_step_forward(a, r_dim, b_dim)
            ddpg.store_transition(s, a, r / 10, s_)
            prev_state = s
            s = s_
            # learn
            if ddpg.pointer == ddpg.memory_capacity:
                print("start learning")
            if ddpg.pointer > ddpg.memory_capacity:
                ddpg.learn()
                if CHANGE:
                    r_var *= .99999
                    b_var *= .99999
            # replace the state
            #s = s_
            # sum up the reward
            ep_reward[episode] += r
            
        # end the episode
        if SCREEN_RENDER:
            env.canvas.tk.destroy()
        status = ddpg.get_status()
        # Print the transitions
        # for transition in status['transitions']:
        #     print("State:", transition['state'])
        #     print("Action:", transition['action'])
        #     print("Reward:", transition['reward'])
        #     print("Next State:", transition['next_state'])
        #     print("------------------------")
                    # Store the transition
        transition = {
            'episode':episode + 1,
            'offload':prev_state[r_dim + b_dim:o_dim].tolist(),
            'bandwidth':prev_state[r_dim:r_dim + b_dim].tolist(),
            'action': a.tolist(),
            'reward': float(r),
            'next_bandwidth': s_[r_dim:r_dim + b_dim].tolist(),
            'next_offload': s_[r_dim + b_dim:o_dim].tolist()
        }
        transitions.append(transition)
        var_counter += 1
        episode += 1
        print(episode)
    
    # Save transitions to a CSV file
    csv_file_path = 'transitions.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(transitions[0].keys())
        # Write the data rows
        for transition in transitions:
            writer.writerow(transition.values())
    print(f"Transitions saved")
    end_time = time.time()
    # Calculate the duration in seconds
    duration = end_time - start_time
    # Print the duration
    print("Runtime: {:.2f} minutes".format(duration / 60))



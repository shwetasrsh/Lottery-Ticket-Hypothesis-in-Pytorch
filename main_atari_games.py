# Importing Libraries for main.py from LTH
import argparse
import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import seaborn as sns
import torch.nn.init as init
import pickle

# Custom Libraries for main.py for LTH
import utils


# Importing Libraries for dqn.py from minimalRL
import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



#for results - we have to draw a graph of comp[_ite] Vs bestacc[_ite]



# Tensorboard initialization  - main.py
writer = SummaryWriter()

# Plotting Style - main.py
sns.set_style('darkgrid')


#Hyperparameters
#learning_rate = 0.0005
#gamma         = 0.98
#buffer_limit  = 50000
batch_size    = 32


#Hyperparameters from open ai gym
learning_rate = 0.001
gamma         = 0.98
buffer_limit  = 50000
#batch_size    = 128
#batch_size = 60 #(LTH)

# dqn.py file
# this buffer is a dataset of our agent's past experiences
# this ensures that the agent is learning from its entire history
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

    
# dqn.py file
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        #self.fc1 = nn.Linear(4, 128)
        self.fc1 = nn.Linear(3, 128)
        #Here 4 because each state representation is an input and that takes 4 preprocessed image frames
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()
            

# dqn.py file            
def train(q, model, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        #s,a,r,s_prime,done_mask = memory.sample(32)
        q_out = q(s)
        #q_a = q_out.gather(1,a)
        q_a = q_out.gather(1,a.unsqueeze(1))
        #q_a = q_a.squeeze(1)
        max_q_prime = model(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# q_target is replaced by model
# Main - main.py
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #reinit = True if args.prune_type=="reinit" else False
    reinit = False
    # Data Loader
    env = gym.make('MsPacman-v0')
    # Architecture
    q = Qnet()
    global model
    #q_target = Qnet()
    model = Qnet()
    model.to(device)
    model.load_state_dict(q.state_dict())
    #q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    # here we have to work with q_target
    
    
    # code from LTH
    # Weight Initialization
    # here model will be q_target or q
    #model.apply(weight_init)
   
    model.apply(weight_init)
    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    # Making Initial Mask
    make_mask(model)
    # Optimizer and Loss  (not necessary, is there in the train method)
    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    # criterion = nn.CrossEntropyLoss() # Default was F.nll_loss
    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    # bestacc = 0.0
    # best_accuracy = 0
    #ITERATION = args.prune_iterations
    ITERATION = 10
    #comp = np.zeros(ITERATION,float)
    #bestacc = np.zeros(ITERATION,float)
    step = 0
    #all_loss = np.zeros(args.end_iter,float)
    #all_accuracy = np.zeros(args.end_iter,float)

    for _ite in range(0, ITERATION):
    #for _ite in range(args.start_iter, ITERATION):
        print('Hello')
        if not _ite == 0:
            #prune_by_percentile(args.prune_percent, resample=resample, reinit=reinit)
            prune_by_percentile(25, resample=resample, reinit=reinit)
            if reinit:
                model.apply(weight_init)
                step = 0
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        weight_dev = param.device
                        param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
                        step = step + 1
                step = 0
            else:
                original_initialization(mask, initial_state_dict)
            optimizer = torch.optim.Adam(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
        #print(f"\n--- Pruning Level [{ITE}:{_ite}/{ITERATION}]: ---")
    #output           
    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = env.reset()
        # reset function returns an initial observation
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)

            #s_prime, r, done, info = env.step(a)
            s_prime, r, done, info = env.step(env.action_space.sample())
            # s_prime => an environment specific object representing your observation of the environment
            # r => amount of reward achieved by the previous action. The scale varies between environments, but the goal is
            # always to increase your total reward
            # done => whether its time to reset the environment again. Most tasks are divided up into well defined episodes 
            # and done being true indicates the episode has terminated.
            # info => diagnostic information useful for debugging. 
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break
            
        if memory.size()>2000:
            train(q, model, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            model.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()




        
        
        
# Prune by Percentile module
def prune_by_percentile(percent, resample=False, reinit=False,**kwargs):
        global step
        global mask
        global model

        # Calculate percentile value
        step = 0
        for name, param in model.named_parameters():

            # We do not prune bias term
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), percent)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])
                
                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                mask[step] = new_mask
                step += 1
        step = 0
       

      
# Function to make an empty mask of the same size as the model
def make_mask(model):
    global step
    global mask
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            step = step + 1
    mask = [None]* step 
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0

    
    
def original_initialization(mask_temp, initial_state_dict):
    global model
    
    step = 0
    for name, param in model.named_parameters(): 
        if "weight" in name: 
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0

    
    
# Function for Initialization
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
                
                
                
if __name__=="__main__":
    
    #from gooey import Gooey
    #@Gooey      
    
    # Arguement Parser
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--lr",default= 1.2e-3, type=float, help="Learning rate")
    #parser.add_argument("--batch_size", default=60, type=int)
    #parser.add_argument("--start_iter", default=0, type=int)
    #parser.add_argument("--end_iter", default=5, type=int)
    #parser.add_argument("--print_freq", default=1, type=int)
    #parser.add_argument("--valid_freq", default=1, type=int)
    #parser.add_argument("--resume", action="store_true")
    #parser.add_argument("--prune_type", default="lt", type=str, help="lt | reinit")
    #parser.add_argument("--gpu", default="0", type=str)
    #parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    #parser.add_argument("--arch_type", default="fc1", type=str, help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")
    #parser.add_argument("--prune_percent", default=10, type=int, help="Pruning percent")
    #parser.add_argument("--prune_iterations", default=1, type=int, help="Pruning iterations count")

    
    #args = parser.parse_args()


    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    #os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    
    #FIXME resample
    resample = False

    # Looping Entire process
    #for i in range(0, 5):
    main()


# to plot ticket reward Vs fraction of weights pruned
# so here we should have averaged episodic reward Vs prune percent(which will be the Compression rate)
# ticket reward => averaged episodic reward over the last L game episodes
# For each game we plot ticket reward curves for both winning and random tickets as the fraction of weight pruned increases

# for plotting two or more lines on the same plot
# x1 => Fraction of weights pruned - random ticket
# y1 => Ticket reward - random ticket
# x2 => Fraction of weights pruned - winning ticket
# y2 => Ticket reward - winning ticket
# plt.plot(x1, y1, label = "line 1")    
# plt.plot(x2, y2, label = "line 2") 
# plt.xlabel('x - axis')  
# plt.ylabel('y - axis') 
# plt.title('Ticket Reward Vs Fraction of weights pruned') 
# plt.legend() 
# plt.show() 















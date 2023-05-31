'''DLP DQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from atari_wrappers import wrap_deepmind, make_atari

# def stack_frame(state_q, k=1):
#     ## 1TODO ##
#     for i in range(len(state_q)-k):
#         state = np.expand_dims(np.squeeze(state_q.popleft()), axis=0)
#         if i == 0:
#             s = state
#         else:
#             s = np.vstack([s, state])
#     #print("stack_frame:",s.shape,type(s))
#     return s


class ReplayMemory(object):
    ## 1TODO ##
    __slot__ = ['buffer']
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        """Saves a transition"""
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        """Sample a batch of transitions"""
        #state, action, reward,  done
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(np.array(x), dtype=torch.float, device=device)
                for x in zip(*transitions))

    def __len__(self):
        return self.size


class Net(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(Net, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(True)
                                        )
        self.classifier = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        #print(x.shape)
        x = x.to('mps')
        x = x.float() / 255.
        x = self.cnn(x)
        #print(x.shape)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)


class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        self._target_net.eval()
        self._optimizer = torch.optim.Adam(self._behavior_net.parameters(), lr=args.lr, eps=1.5e-4)

        ## 1TODO ##
        """Initialize replay buffer"""
        #self._memory = ReplayMemory(...)
        self._memory = ReplayMemory(capacity=args.capacity)
        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        ## 1TODO ##
        # state = stack_frame(state)
        # state = np.expand_dims(state ,axis=0)
        rand = random.random()
        if rand <= epsilon:     # pick random action
            action = action_space.sample()
        else:                   # pick argmax(Q(s,a)), given state predict action
            with torch.no_grad():
                #print("st",state.shape)
                state = np.array(state)

                vec = self._behavior_net.forward(torch.from_numpy(state).unsqueeze(0).permute(0,3,1,2).to(self.device))
                #print("vec",vec.shape)
                action = torch.argmax(vec).item()
                #print(action)
        return action
        

    def append(self, state, action, reward, next_state,done):
        ## 1TODO ##
        """Push a transition into replay buffer"""
        #self._memory.push(...)
        # state = stack_frame(state, k=0)
        #print("append",state.shape)
        self._memory.push(state, [action], [reward], next_state,[int(done)])
        #data augmentation
        # if action == 2:
        #     self._memory.push(np.flip(state, axis=2), [3], [reward], [int(done)])
        # elif action == 3:
        #     self._memory.push(np.flip(state, axis=2), [2], [reward], [int(done)])
        # else:
        #     self._memory.push(np.flip(state, axis=2), [action], [reward], [int(done)])



    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)
        #print(type(state),type(action),type(reward),type(done))
        #print(len(state),action.shape,reward.shape)
        #print("state shape",state.shape) # torch.Size([B, 5, 84, 84])
        # _, next_state = torch.split(state, split_size_or_sections=[1, 4], dim=1)
        # state, _ = torch.split(state, split_size_or_sections=[4, 1], dim=1)
        #print("spilt",state.shape, next_state.shape)
        ## 1TODO ##
        # q_value, _ = torch.max(self._behavior_net.forward(state), dim=1)
        # q_value = torch.unsqueeze(q_value, dim=1)
        # with torch.no_grad():
        #     vec = self._target_net.forward(next_state)
        #     q_next, _ = torch.max(vec, dim=1)
        #     q_next = torch.unsqueeze(q_next, dim=1)
        #     q_target = reward + gamma * q_next * (1-done)
        state = state.permute(0,3,1,2)
        next_state = next_state.permute(0,3,1,2)
        q_value = self._behavior_net(state)
        q_value = torch.gather(q_value, 1, action.long())
        with torch.no_grad():
            vec = self._behavior_net(next_state)
            action = torch.argmax(vec, dim=1)
            vec = torch.gather(vec, 1, action.view([self.batch_size, 1]).long())
            vec = reward + gamma * vec * (1-done)

        
        #print(q_value.shape,q_target.shape)
        criterion = nn.MSELoss()
        loss = criterion(q_value, vec) #
        
        # # optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()            

    def soft_update(self, behavior_model, target_model, tau=1e-3):
        for target_param, behavior_param in zip(target_model.parameters(), behavior_model.parameters()):
            target_param.data.copy_(tau*behavior_param.data + (1.0-tau)*target_param.data)
        

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        ## 1TODO ##
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, model_path, checkpoint=True):
        if checkpoint:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
                'target_net': self._target_net.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=True):
        model = torch.load(model_path, map_location="mps")
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])

def train(args, agent, writer):
    print('Start Training')
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env_raw, episode_life=True, clip_rewards=True, frame_stack=True, scale=True)
    action_space = env.action_space
    total_steps, epsilon = 0, 0.1
    ewma_reward = 0

    #state_q = deque(maxlen=5)
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        #state_q.clear()
        # for i in range(4):
        #     state_q.append(state)
        state, reward, done, _ = env.step(1) # fire first !!!
        #state_q.append(state)
        for t in itertools.count(start=1):
            #print("t",len(state_q))
            # if t%100 == 0:
            #       time.sleep(1)

            # if total_steps < args.warmup:
            #     action = action_space.sample()
            # else:
                # select action
            action = agent.select_action(state, epsilon, action_space)
            # decay epsilon
            epsilon -= (1 - args.eps_min) / args.eps_decay
            epsilon = max(epsilon, args.eps_min)

            #### execute action
            next_state, reward, done, _ = env.step(action)
            #state_q.append(state)
            ## 1TODO ##
            # store transition
            #agent.append(...)
            agent.append(state, action, reward, next_state,done)

            if total_steps >= args.warmup:
                agent.update(total_steps)

            total_reward += reward
            state = next_state
            # TODO #
            if total_steps % args.eval_freq == 0 and episode != 0 :
                """You can write another evaluate function, or just call the test function."""
                tmp_score = test(args, agent, writer)
                agent.save(args.model + "dqn_" + str(total_steps) +"_"+str(tmp_score)+ ".pt")

            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward, episode)
                writer.add_scalar('Train/Ewma Reward', ewma_reward, episode)
                print('Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                        .format(total_steps, episode, t, total_reward, ewma_reward, epsilon))
                break
    env.close()


def test(args, agent, writer):
    print('Start Testing')
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env_raw, episode_life=False, clip_rewards=False)
    action_space = env.action_space
    e_rewards = []
    #state_q = deque(maxlen=5)
    for i in range(args.test_episode):
        state = env.reset()
        # state_q.clear()
        # for k in range(5):
        #     state_q.append(state)
        state, reward, done, _ = env.step(1) # fire first !!!
        e_reward = 0
        done = False

        while not done:
            time.sleep(0.01)
            env.render()
            action = agent.select_action(state, args.test_epsilon, action_space)
            next_state, reward, done, _ = env.step(action)
            state=next_state
            e_reward += reward

        print('episode {}: {:.2f}'.format(i+1, e_reward))
        e_rewards.append(e_reward)

    env.close()
    print('Average Reward: {:.2f}'.format(float(sum(e_rewards)) / float(args.test_episode)))
    return float(sum(e_rewards)) / float(args.test_episode)

def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='mps')
    parser.add_argument('-m', '--model', default='')
    parser.add_argument('--logdir', default='log/dqn')
    # train
    parser.add_argument('--warmup', default=50000, type=int) #20000
    parser.add_argument('--episode', default=1000000, type=int) #20000
    parser.add_argument('--capacity', default=100000, type=int) #100000
    parser.add_argument('--batch_size', default=64, type=int) #32
    parser.add_argument('--lr', default=0.0000625, type=float)
    parser.add_argument('--eps_decay', default=1000000, type=float)
    parser.add_argument('--eps_min', default=0.1, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=10000, type=int) #10000
    parser.add_argument('--eval_freq', default=100000, type=int) #200000
    # test
    # Remember to set episode_life=False, clip_rewards=False while testing
    parser.add_argument('--test_only', default=True, action='store_true')
    parser.add_argument('-tmp', '--test_model_path', default='ckpt/dqn_1000000.pt')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--test_episode', default=10, type=int)
    parser.add_argument('--seed', default=20230422, type=int)
    parser.add_argument('--test_epsilon', default=0.0002, type=float)
    args, unknown = parser.parse_known_args()
    ## main ##
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    if args.test_only:
        agent.load("dqn_316.5.pt")
        test(args, agent, writer)
    else:
        agent.load(model_path="/Users/cheng/Desktop/bk_final_fast/dqn_1800000_229.5.pt")
        train(args, agent, writer)
    


if __name__ == '__main__':
    main()

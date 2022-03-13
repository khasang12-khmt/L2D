from mb_agg import *
from agent_utils import *
import torch
from Params import configs
import time
import numpy as np


device = torch.device(configs.device)

N_JOBS_P = 6
N_MACHINES_P = 6
LOW = 1
HIGH = 99
SEED = 200
N_JOBS_N = 6
N_MACHINES_N = 6


from JSSP_Env import SJSSP
from PPO_jssp_multiInstances import PPO


dataLoaded = np.load('/content/L2D/DataGen/Vali/generatedData10_10_Seed200.npy')
_, N_JOBS_P, N_MACHINES_P = dataLoaded[0].shape
_, N_JOBS_N, N_MACHINES_N = dataLoaded[0].shape
env = SJSSP(n_j=N_JOBS_P, n_m=N_MACHINES_P)

ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
          n_j=N_JOBS_P,
          n_m=N_MACHINES_P,
          num_layers=configs.num_layers,
          neighbor_pooling_type=configs.neighbor_pooling_type,
          input_dim=configs.input_dim,
          hidden_dim=configs.hidden_dim,
          num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
          num_mlp_layers_actor=configs.num_mlp_layers_actor,
          hidden_dim_actor=configs.hidden_dim_actor,
          num_mlp_layers_critic=configs.num_mlp_layers_critic,
          hidden_dim_critic=configs.hidden_dim_critic)
path = './SavedNetwork/{}.pth'.format(str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(LOW) + '_' + str(HIGH))
# ppo.policy.load_state_dict(torch.load(path))
ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
# ppo.policy.eval()
g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                         batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                         n_nodes=env.number_of_tasks,
                         device=device)




dataset = []

for i in range(1):
    dataset.append((dataLoaded[i][0], dataLoaded[i][1]))



def test(dataset):
    result = []
    # torch.cuda.synchronize()
    t1 = time.time()
    for i, data in enumerate(dataset):
        adj, fea, candidate, mask = env.reset(data)
        ep_reward = - env.max_endTime
        # delta_t = []
        # t5 = time.time()
        while True:
            # t3 = time.time()
            fea_tensor = torch.from_numpy(fea).to(device)
            adj_tensor = torch.from_numpy(adj).to(device)
            candidate_tensor = torch.from_numpy(candidate).to(device)
            mask_tensor = torch.from_numpy(mask).to(device)
            # t4 = time.time()
            # delta_t.append(t4 - t3)

            with torch.no_grad():
                pi, _ = ppo.policy(x=fea_tensor,
                                   graph_pool=g_pool_step,
                                   padded_nei=None,
                                   adj=adj_tensor,
                                   candidate=candidate_tensor.unsqueeze(0),
                                   mask=mask_tensor.unsqueeze(0))
                # action = sample_select_action(pi, omega)
                action = greedy_select_action(pi, candidate)

            adj, fea, reward, done, candidate, mask = env.step(action)
            ep_reward += reward

            if done:
                
                break
        # t6 = time.time()
        # print(t6 - t5)
        # print(max(env.end_time))
    print('Makespan:', -ep_reward + env.posRewards)
    # print(sum(delta_t))
    # torch.cuda.synchronize()
    t2 = time.time()
    print(t2 - t1)

test(dataset)
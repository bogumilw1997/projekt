from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from json import load
from tqdm import tqdm
import networkx as nx

# r0 = 0.8 dla mu=lambda 

plt.rcParams["figure.figsize"] = [14, 8]
plt.rcParams['font.size'] = '15'
plt.rcParams['lines.linewidth'] = '1.5'
plt.rcParams['lines.markersize'] = '9'
plt.rcParams["figure.autolayout"] = True

with open("semestr3\Wdpr\projekt\parallel\params.json") as f:
    parameters = load(f)
    
N = parameters['N']
m = parameters['m'] 
T = parameters['T']

epsilon = parameters['epsilon']

mu = parameters['mu'] # prawd. wyzdr.
gamma = parameters['gamma']
realizations = parameters['realizations']
eta = parameters["eta"]
p0 = parameters["p0"]
integration_steps = parameters["integration_steps"]
lambd = parameters["lambd"]
save_path_local = parameters["save_path_local"]
processes = parameters['processes']

df = pd.DataFrame()
df_ = pd.DataFrame()
    
nodes_list = np.arange(N)

epidemic_states = np.array(['S', 'I'])

epidemic_state = np.random.choice(epidemic_states, N, p=[1- p0, p0])

g = nx.barabasi_albert_graph(N, m)

def check_epidemic(node):
        
        if epidemic_state[node] == 'S':
            
            neighbours = np.array(g[node])
            
            if neighbours.shape[0] > 0:
                    
                    i_neighbours = neighbours[epidemic_state[neighbours] == 'I']
                    
                    i_count = i_neighbours.shape[0]
                    
                    if i_count > 0:
                        
                        infection_prob = np.random.rand(i_count)
                        
                        if all(i_p >= lambd for i_p in infection_prob):
                            return 'S'
                        else:
                            return 'I'
                    else:
                        return 'S'
            else:
                return 'S'
                            
        elif epidemic_state[node] == 'I':
            
            recovery_prob = np.random.rand()
            
            if recovery_prob <= mu:
                return 'S'
            else:
                return 'I'
      
def get_inf_number():
    
    return np.sum(epidemic_state == "I")/N

def get_susc_number():

    return np.sum(epidemic_state == "S")/N
    
def init_pool(es, m, gg, lamb):
    global epidemic_state
    global mu
    global g
    global lambd

    epidemic_state = es
    mu = m
    g = gg
    lambd = lamb 

if __name__ == '__main__':
    
    infections_list = np.zeros(T)
    susc_list = np.zeros(T)
    
    infections_list[0] = get_inf_number()
    susc_list[0] = get_susc_number()

    
    for t in tqdm(range(1, T)):
        
        with ProcessPoolExecutor(max_workers=processes, initializer=init_pool, initargs=(epidemic_state,mu,g,lambd)) as executor:
            r = executor.map(check_epidemic, nodes_list)
        
        epidemic_state = np.array(list(r))

        infections_list[t] = get_inf_number()
        susc_list[t] = get_susc_number()
        
    df_['inf'] = infections_list
    df_['susc'] = susc_list
    
    df_['lambd'] = lambd
    df_['t'] = np.arange(T)
    
    df = pd.concat([df, df_], ignore_index = True)
    
    df.to_csv(save_path_local + 'si_test1.csv')

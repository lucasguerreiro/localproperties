from util import rw, rwd, rwid, tsaw, tsaw_node
import networkx as nx
import igraph
import pickle
import xnet

models_list = ['er', 'ba', 'wax', 'lfr', 'facebook', 'power', 'econ-poli', 'web-EPA', 'Collins', 'lfr_m05', 'AI_interactions', 'socfb-JohnsHopkins55', 'lfr_m8']
#realnets_list = ['econ-poli', 'web-EPA', 'Collins'] #4119, 8909, 9070

dynamics_list = ['rw', 'rwd', 'rwid', 'tsaw', 'tsaw_node']

TOTAL_NETS = 1 # numero de redes que foram utilizadas no gen_nets
WALK_SIZE = 50000 # tamanho maximo da caminhada
TOTAL_WALKS = 20 # numero de caminhadas a serem executadas e gravadas

n = 5000
k = 4


# load recorded nets
def load_net(model, n, k, net_number):

    net = nx.read_edgelist('net_' + model + '_' + str(n) + '_' + str(k) + '_' + str(net_number) + '.edgeslist')
    return net

"""
def load_real_net(model):

    n = 5000
    k = 4
    net_number = 0

    net = nx.read_edgelist('net_' + model + '_' + str(n) + '_' + str(k) + '_' + str(net_number) + '.edgeslist')

    return net
"""

def run_dynamics(net, walk_size, total_walks):

    all_walks = []

    for t in range(total_walks):

        walks = {}

        for dynamics in dynamics_list:

            if dynamics == 'rw':
                r = rw(net, walk_size)
            if dynamics == 'rwd':
                r = rwd(net, walk_size)
            if dynamics == 'rwid':
                r = rwid(net, walk_size)
            if dynamics == 'tsaw':
                r = tsaw(net, walk_size)
            if dynamics == 'tsaw_node':
                r = tsaw_node(net, walk_size)

            walks[dynamics] = r

        all_walks.append(walks)

    return all_walks

with open('walks2.pkl', 'rb') as f:
    final_walks = pickle.load(f)

for model in models_list:

    if final_walks.get(model) is None:
        final_walks[model] = {}

    for m in range(TOTAL_NETS):

        if final_walks[model].get(m) is None:

            print('performing', model, str(n), str(k), str(m))

            net = load_net(model, n, k, m)
            walks = run_dynamics(net, WALK_SIZE, TOTAL_WALKS)

            final_walks[model][m] = walks

with open('walks2.pkl', 'wb') as f:
    pickle.dump(final_walks, f)
import networkx as nx
import igraph

from util import build_er, build_ba, build_wax, build_lfr, build_facebook, build_power, build_real

# for each config:
TOTAL_NETS = 1

#models_list = ['econ-poli', 'web-EPA', 'Collins'] #['er', 'ba', 'wax', 'lfr', 'facebook', 'power', ''lfr_m05', 'AI_interactions', 'socfb-JohnsHopkins55']
models_list = ['lfr_m8']
k_list = [4]

params = {'n': 5000,
          'k': 4,
          'options':
              {
                'save_net': True
              }
          }


def load_net(model, params):
    if model == 'er':
        net = build_er(params['n'], params['k'])
    if model == 'ba':
        net = build_ba(params['n'], params['k'])
    if model == 'wax':
        net = build_wax(params['n'], params['k'])
    if model == 'lfr':
        net = build_lfr(params['n'], params['k'], m=0.2)
    if model == 'lfr_m05':
        net = build_lfr(params['n'], params['k'], m=0.05)
    if model == 'lfr_m8':
        net = build_lfr(params['n'], params['k'], m=0.8)
    if model == 'facebook':
        net = build_facebook()
    if model == 'power':
        net = build_power()
    if model == 'econ-poli':
        net = build_real(model)
    if model == 'web-EPA':
        net = build_real(model)
    if model == 'Collins':
        net = build_real(model)
    if model == 'AI_interactions':
        net = build_real(model)
    if model == 'socfb-JohnsHopkins55':
        net = build_real(model)


    if params['options']['save_net']:
        print('saving net {} {} {} {}'.format(model, str(params['n']), str(params['k']), str(params['model_number'])))
        with open('net_' + model + '_' + str(params['n']) + '_' + str(params['k']) + '_' + str(params['model_number']) + '.edgeslist', 'wb') as f:
            nx.write_edgelist(net, f)

    return net

"""
gerar maior componente:
for m in models_list:
    net = load_net(m, 5000, 4, 0)
    giant = max((net.subgraph(c) for c in nx.connected_components(net)), key=len)
    with open('gnet_' + m + '_' + '5000' + '_' + '4' + '_' + '0' + 'edgeslist', 'wb') as f:
        nx.write_edgelist(giant, f)
        
"""

def load_nets(number_nets, model):

    nets_loaded = []

    n = 5000

    for ne in range(number_nets):

        for k in k_list:

            params['k'] = k
            params['model_number'] = ne

            description_net = (model, n, k, ne)

            print(description_net)

            nets_loaded.append((load_net(model, params), description_net))

    return nets_loaded


for model in models_list:

    print('generating ' + model)
    nets = load_nets(TOTAL_NETS, model)
    print('generated ' + model)
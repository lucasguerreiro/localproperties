import pickle
from re import T
import networkx as nx
import igraph
import os.path
import xnetwork as xn
from datetime import datetime
import json


# estava dando erro de pickle:
#from multiprocessing import Pool
# nova tentativa:
from multiprocessing.pool import ThreadPool as Pool


from itertools import product
import multiprocessing
from tqdm.auto import tqdm
import os

# from clusim.clustering import Clustering, print_clustering

import leidenalg

models_list = ['er', 'ba', 'wax', 'lfr', 'facebook', 'power', 'econ-poli', 'web-EPA', 'bio-DM-CX', 'lfr_m05', 'AI_interactions', 'socfb-JohnsHopkins55', 'lfr_m8']
dynamics_list = ['rw', 'rwd', 'rwid', 'tsaw', 'tsaw_node']

WALK_SIZES_LIST = [100, 200, 400, 500, 600, 800, 1000, 2000, 5000, 20000, 50000]
# WALK_SIZES_LIST = [50000]


TOTAL_NETS = list(range(1))
TOTAL_WALKS = list(range(20))

n = 5000
k = 4
properties_list = ['degree', 'closeness', 'eccentricity', 'cc', 'coreness', 'knowledge', 'community_leiden','betweenness']


all_walks = None

def load_net(model, net_number):

    net = nx.read_edgelist('gnet_' + model + '_' + str(n) + '_' + str(k) + '_' + str(net_number) + '.edgeslist')
    return net

def reconstruct_net(walk):
    # reconstruir rede a partir de caminhada

    start_node = str(walk[0])

    nodesIndices = sorted(list(set(walk)))
    node2Index = dict(zip(nodesIndices,range(len(nodesIndices))))
    edges = list(zip(walk[:-1], walk[1:]))
    edgesIndices = [(node2Index[fromIndex], node2Index[toIndex]) for fromIndex,toIndex in edges]

    G = igraph.Graph(len(nodesIndices), edges=list(set(edgesIndices)), directed=False).simplify()
    G.vs['originalIndex'] = nodesIndices

    return G

def get_node_degree(G, v):
    return G.vs["degree"][v]

def get_node_betweenness(G, v):
    return G.vs["betweenness"][v]

def get_node_closeness(G, v):
    return G.vs["closeness"][v]

def get_node_eccentricity(G, v):
    return G.vs["eccentricity"][v]

def get_total_vertices(G):
    return G.vcount()

def get_coreness(G, v):
    return G.vs["coreness"][v]

#https://igraph.org/python/doc/api/igraph._igraph.GraphBase.html#transitivity_local_undirected
def get_node_cc(G, v):
    cc = G.vs["cc"][v]
    if(cc==cc):
        return cc
    else: #nan
        return 0

    
def get_community_ml(membership, v):

    return membership[v]

def get_community_leiden(membership, v):

    return membership[v]


def run_props(args):

    # prop, model, dynamics, walk_size, net_number, walk_number
    prop = args[0]
    model = args[1]
    walk_size = args[2]
    net_number = args[3]

    filename = '_'.join(('sg_wnew_', model, str(net_number), str(walk_size), prop))

    #file_exists = os.path.exists(filename+'.pkl')
    #if file_exists:
    #    return

    # ignorando essa verificacao acima, pois esse arquivo
    # quer apenas recuperar e salvar edgeslist
    # das caminhadas reconstruidas

    print('starting', model, prop)

    net = load_net(model, net_number)
    walks = all_walks[model][net_number]

    # converting net to igraph
    # nx.write_graphml(net, filename + 'temp.graphml')
    # G_original = igraph.read(filename + 'temp.graphml', format="graphml")
    # os.remove(filename + 'temp.graphml')
    G_original = igraph.Graph.from_networkx(net)
    
    G_original.vs["originalIndex"] = G_original.vs["_nx_name"]

    originalName2Index = dict(zip(G_original.vs["originalIndex"], range(G_original.vcount())))

    full_result = {}
    # xn.igraph2xnet(G_original,"networks/"+filename+"_original.xnet")
    
    for walk_number, walk in enumerate(walks):

        print(model, prop, str(walk_number)," "*20,end="\r")

        walk_result = {}

        for dynamics in dynamics_list:

            G = reconstruct_net(walk[dynamics][:walk_size])

            # salvar edgeslist aqui

            if prop == 'community_ml':
                membership_rec = G.community_multilevel().membership
                membership_orig = G_original.community_multilevel().membership
            
            if prop == 'community_leiden':
                membership_rec = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition).membership
                membership_orig = leidenalg.find_partition(G_original, leidenalg.ModularityVertexPartition).membership
                G_original.vs["communities"]=membership_orig
                xn.igraph2xnet(G_original,filename+"_original.xnet")
                G.vs["communities"]=membership_rec
                xn.igraph2xnet(G,filename+"_%d_%s.xnet"%(walk_size,dynamics))
            
            result = {}
            
            for v_sequence, v in enumerate(G.vs()):

                v_name = v["originalIndex"]

                index_reconstructed = v_sequence # G.vs._name_index[v_name]
                index_original = originalName2Index[v_name] #G_original.vs.find(id=v_name).index  # G_original.vs._name_index[v_name]

                r_graph = {}
                o_graph = {}
                
                if prop == 'degree':
                    if("degree" not in G_original.vertex_attributes()):
                        G_original.vs["degree"] = G_original.degree()
                    if("degree" not in G.vertex_attributes()):
                        G.vs["degree"] = G.degree()
                    r_graph['degree'] = get_node_degree(G, index_reconstructed)
                    o_graph['degree'] = get_node_degree(G_original, index_original)

                    point = (r_graph['degree'], o_graph['degree'])

                if prop == 'betweenness':
                    if("betweenness" not in G_original.vertex_attributes()):
                        G_original.vs["betweenness"] = G_original.betweenness()
                    if("betweenness" not in G.vertex_attributes()):
                        G.vs["betweenness"] = G.betweenness()
                    r_graph['betweenness'] = get_node_betweenness(G, index_reconstructed)
                    o_graph['betweenness'] = get_node_betweenness(G_original, index_original)

                    point = (r_graph['betweenness'], o_graph['betweenness'])

                if prop == 'closeness':
                    if("closeness" not in G_original.vertex_attributes()):
                        G_original.vs["closeness"] = G_original.closeness()
                    if("closeness" not in G.vertex_attributes()):
                        G.vs["closeness"] = G.closeness()
                    r_graph['closeness'] = get_node_closeness(G, index_reconstructed)
                    o_graph['closeness'] = get_node_closeness(G_original, index_original)

                    point = (r_graph['closeness'], o_graph['closeness'])

                if prop == 'eccentricity':
                    if("eccentricity" not in G_original.vertex_attributes()):
                        G_original.vs["eccentricity"] = G_original.eccentricity()
                    if("eccentricity" not in G.vertex_attributes()):
                        G.vs["eccentricity"] = G.eccentricity()
                    r_graph['eccentricity'] = get_node_eccentricity(G, index_reconstructed)
                    o_graph['eccentricity'] = get_node_eccentricity(G_original, index_original)

                    point = (r_graph['eccentricity'], o_graph['eccentricity'])

                if prop == 'cc':
                    if("cc" not in G_original.vertex_attributes()):
                        G_original.vs["cc"] = G_original.transitivity_local_undirected(mode=0)
                    if("cc" not in G.vertex_attributes()):
                        G.vs["cc"]  = G.transitivity_local_undirected(mode=0)
                    r_graph['cc'] = get_node_cc(G, index_reconstructed)
                    o_graph['cc'] = get_node_cc(G_original, index_original)
                    
                    point = (r_graph['cc'], o_graph['cc'])

                if prop == 'knowledge':

                    r_graph['knowledge'] = get_total_vertices(G)
                    o_graph['knowledge'] = get_total_vertices(G_original)

                    point = (r_graph['knowledge'], o_graph['knowledge'])

                if prop == 'coreness':
                    if("coreness" not in G_original.vertex_attributes()):
                        G_original.vs["coreness"] = G_original.coreness()
                    if("coreness" not in G.vertex_attributes()):
                        G.vs["coreness"]  = G.coreness()
                    r_graph['coreness'] = get_coreness(G, index_reconstructed)
                    o_graph['coreness'] = get_coreness(G_original, index_original)

                    point = (r_graph['coreness'], o_graph['coreness'])

                if prop == 'community_ml':
                    r_graph['community_ml'] = get_community_ml(membership_rec, index_reconstructed)
                    o_graph['community_ml'] = get_community_ml(membership_orig, index_original)

                    point = (r_graph['community_ml'], o_graph['community_ml'])

                if prop == 'community_leiden':
                    r_graph['community_leiden'] = get_community_leiden(membership_rec, index_reconstructed)
                    o_graph['community_leiden'] = get_community_leiden(membership_orig, index_original)

                    point = (r_graph['community_leiden'], o_graph['community_leiden'])

                result[v_sequence] = point
                print(point)

            walk_result[dynamics] = result

        full_result[walk_number] = walk_result

    with open("data/"+filename + '.pkl', 'wb') as f:
        pickle.dump(full_result, f)

    print('finishing', model, prop)




if __name__ == '__main__':

    with open('walks_new_g.pkl', 'rb') as f:
        all_walks = pickle.load(f)

    permlist = list(product(*[properties_list, models_list, WALK_SIZES_LIST, TOTAL_NETS]))

    num_processors = multiprocessing.cpu_count()

    pool = Pool(processes=num_processors)

    for result in tqdm(pool.imap(func=run_props, iterable=permlist), total=len(permlist)):
        pass
    pool.close()





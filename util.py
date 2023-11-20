import networkx as nx
from igraph import *
import xnet
import random

import scipy.spatial.distance as dist

import numpy as np

import subprocess
from subprocess import PIPE

import graph_tool.all as gt

def build_er(n, k):

    r = k / n
    szg = 0
    nodes_er = int(n*1.01)
    while (szg < n):
        er_u = nx.erdos_renyi_graph(nodes_er, r, directed=False)
        net = max((er_u.subgraph(c) for c in nx.connected_components(er_u)), key=len)
        szg = net.number_of_nodes()
        nodes_er += 1

    return net

def build_ba(n, k):

    r = k / 2
    net = nx.barabasi_albert_graph(n, int(r))

    return net

def build_wax(n, k):

    list_ab_wax = {500: [(0.32, 0.051), (0.4, 0.057), (0.44, 0.063), (0.59, 0.06)],
                   1000: [(0.45, 0.029), (0.47, 0.035), (0.48, 0.041), (0.43, 0.047)],
                   5000: [(0.18, 0.02), (0.27, 0.02), (0.25, 0.024), (0.19, 0.031)]}

    ref_list = {4: 0, 6: 1, 8: 2, 10: 3}

    ab = list_ab_wax[n][ref_list[k]]
    d = 0
    while not (k - 0.1 <= d <= k + 0.1):
        igraph_net = Waxman1(n, alpha=ab[0], beta=ab[1])
        d = np.mean(igraph_net.degree())

    net = nx.Graph(igraph_net.get_edgelist())

    return net


def Waxman1(N,alpha=1.,beta=0.015):

    pos = np.random.rand(N,2)								# Posicoes aleatorias entre 0 e 1
    D = dist.pdist(pos)
    D = dist.squareform(D)									# Matrix de distancias entre os nos

    L = np.max(D)
    Pc = alpha*np.exp(-D/float(beta*L))						# Probabilidades

    rand_vals = np.random.rand(N,N)
    A = rand_vals < Pc										# Matriz de adjacencia

    g = Graph.Adjacency(A.tolist(),mode=ADJ_UPPER)
    g.simplify()

    return g

def build_lfr(n, k, m=0.2, version=1):

    #version 1 = benchmark/xnet implementation
    #version 2 = nx implementation

    # parametros fixados:
    mixing = m
    nc = 5
    t1 = 3
    t2 = 0

    if version == 1:

        ref_list = {4: 0, 6: 1, 8: 2, 10: 3}

        maxk_list = {0: [7, 11, 15, 19],
                     3: [53, 171, 0, 0],
                     6: [0, 0, 0, 0]}

        if maxk_list[t1][ref_list[k]] > 0:
            maxk = maxk_list[t1][ref_list[k]]
        else:
            maxk = (n / 2)

        p = subprocess.Popen(["./benchmark", "-N", str(n), "-k", str(k), "-maxk", str(maxk), "-minc", str(n / nc), "-maxc",
                              str(n / nc), "-mu", str(mixing), "-t1", str(t1), "-t2", str(t2)], stdin=PIPE, stdout=PIPE) # , bufsize=1)

        a, b = p.communicate()
        #shutil.copy('network.xnet', new_name)

        net_xnet = xnet.xnet2igraph("network.xnet")
        net_xnet = net_xnet.simplify()
        n2 = net_xnet.get_edgelist()

        net = nx.Graph(n2)

    else:
        net = nx.generators.community.LFR_benchmark_graph(n, t1, t2, mixing, average_degree=k, min_community=n/nc, max_community=n/nc)


    return net


def build_facebook():

    g_facebook = gt.collection.ns["facebook_organizations/S1"]
    n2 = list(g_facebook.get_edges())

    net = nx.Graph(n2)
    return net


def build_power():

    g_power = gt.collection.ns["power"]

    n2 = list(g_power.get_edges())

    net = nx.Graph(n2)
    return net

def build_real(model):

    net_xnet = xnet.xnet2igraph("Shared Networks/" + model + ".xnet")
    net_xnet = net_xnet.simplify()
    n2 = net_xnet.get_edgelist()

    net = nx.Graph(n2)


    return net


def rw(network, max_iterations, no1=None):
    listRW = []

    if no1 == None:
        listnodes = list(network.nodes())
        no1 = random.choice(listnodes)

    listRW.append(no1)

    for i in range(max_iterations - 1):
        no2 = random.choice(list(network[no1]))

        listRW.append(no2)
        no1 = no2

    return listRW


def rwd(network, max_iterations, no1=None):
    # probabilities biased toward nodes with higher degrees

    listRWD = []

    if no1 == None:
        listnodes = list(network.nodes())

        no1 = random.choice(listnodes)

    listRWD.append(no1)

    for i in range(max_iterations - 1):

        somadegree = 0
        for j in network[no1]:
            # somadegree += network.degree()[network[no1][j]]
            somadegree += len(network[j])

        somarelativa = 0

        frel = [0] * len(network[no1])

        c = 0
        for j in network[no1]:
            somarelativa += len(network[j]) / somadegree
            frel[c] = somarelativa
            c += 1

        sort = random.random()

        c = 0
        for j in network[no1]:
            if frel[c] > sort:
                no2 = j
                break
            c += 1

        listRWD.append(no2)
        no1 = no2

    return listRWD


def rwid(network, max_iterations, no1=None):
    # probabilities biased toward nodes with higher degrees

    listRWID = []

    if no1 == None:
        listnodes = list(network.nodes())

        no1 = random.choice(listnodes)

    listRWID.append(no1)

    for i in range(max_iterations - 1):

        somadegree = 0
        for j in network[no1]:
            # somadegree += network.degree()[network[no1][j]]
            somadegree += 1 / len(network[j])

        somarelativa = 0

        frel = [0] * len(network[no1])

        c = 0
        for j in network[no1]:
            somarelativa += (1 / len(network[j])) / somadegree
            frel[c] = somarelativa
            c += 1

        sort = random.random()

        c = 0
        for j in network[no1]:
            if frel[c] > sort:
                no2 = j
                break
            c += 1

        listRWID.append(no2)
        no1 = no2

    return listRWID

def tsaw(network, max_iterations, no1=None, directed=False):
    # edges already visited by the agent are avoided

    gama = 2

    listTSAW = []

    listnodes = list(network.nodes())

    if no1 == None:
        no1 = random.choice(listnodes)

    listTSAW.append(no1)

    # visits = np.zeros((nodes + 200, nodes + 200))
    visits = {}

    for i in range(max_iterations - 1):

        somavisits = 0

        for j in network[no1]:
            if visits.get(no1):
                if visits[no1].get(j):
                    somavisits += gama ** (-visits[no1][j])
                else:
                    visits[no1][j] = 0
                    somavisits += gama ** (-visits[no1][j])
            else:
                visits[no1] = {}
                visits[no1][j] = 0
                somavisits += gama ** (-visits[no1][j])

        somarelativa = 0

        frel = [0] * len(network[no1])

        if somavisits == 0:
            somavisits = 1

        c = 0
        for j in network[no1]:
            somarelativa += (gama ** (-visits[no1][j])) / somavisits
            frel[c] = somarelativa
            c += 1

        sort = random.random()

        c = 0
        for j in network[no1]:
            if frel[c] > sort:
                no2 = j
                break
            c += 1

        try:
            visits[no1][no2] += 1
        except:
            visits[no1][no2] = 1

        if not directed:
            if visits.get(no2):
                if visits[no2].get(no1):
                    visits[no2][no1] += 1
                else:
                    visits[no2][no1] = 1
            else:
                visits[no2] = {no1: 1}

        listTSAW.append(no2)
        no1 = no2

    return listTSAW



def tsaw_node(network, max_iterations, no1=None, directed=False):
    # edges already visited by the agent are avoided

    gama = 2

    listTSAW = []

    listnodes = list(network.nodes())

    if no1 == None:
        no1 = random.choice(listnodes)

    listTSAW.append(no1)

    # visits = np.zeros((nodes + 200, nodes + 200))
    visits = {}
    for node in listnodes:
        visits[node] = 0

    for i in range(max_iterations - 1):

        somavisits = 0

        for j in network[no1]:
            somavisits += gama ** (-visits[j])

        somarelativa = 0

        frel = [0] * len(network[no1])

        if somavisits == 0:
            somavisits = 1

        c = 0
        for j in network[no1]:
            somarelativa += (gama ** (-visits[j])) / somavisits
            frel[c] = somarelativa
            c += 1

        sort = random.random()

        c = 0
        for j in network[no1]:
            if frel[c] > sort:
                no2 = j
                break
            c += 1

        visits[no2] += 1

        listTSAW.append(no2)
        no1 = no2

    return listTSAW


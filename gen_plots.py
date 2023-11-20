import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr, spearmanr
import numpy as np
import math

from sklearn import metrics
import clusim.sim as sim
from clusim.clustering import Clustering, print_clustering

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

TOTAL_WALKS = 20
TOTAL_NETS = 1

WALK_SIZE_LIST = [100, 200, 400, 500, 600, 800, 1000, 2000, 5000, 20000, 50000]

models_list_t = ['er', 'wax', 'ba', 'lfr_m05', 'lfr', 'lfr_m8']#,'facebook', 'power', 'econ-poli', 'web-EPA', 'Collins', 'AI_interactions', 'socfb-JohnsHopkins55']
models_list_r = ['facebook', 'power', 'econ-poli', 'web-EPA', 'bio-DM-CX', 'AI_interactions', 'socfb-JohnsHopkins55']
properties_list = ['degree', 'closeness', 'cc', 'eccentricity', 'betweenness', 'coreness'] #community_ml betweenness

models_list_plot_t = ['ER', 'Wax', 'BA', 'LFR,m=0.05', 'LFR,m=0.2', 'LFR,m=0.8']
models_list_plot_r = ['Facebook', 'Power Grid', 'Economics', 'Web (EPA)', 'Bio (DM-CX)', 'Bio (AI)', 'Social (JH)']
properties_list_plot = ['Degree', 'Closeness', 'CC', 'Eccentricity', 'Betweenness', 'Coreness'] #community_ml betweenness

dynamics_list = ['rw', 'rwd', 'rwid', 'tsaw', 'tsaw_node']
dynamics_list_plot = ['RW', 'RWD', 'RWID', 'TSAW_Edge', 'TSAW_Node']
k_list = [4]


def plot_graph1():

    for prop in properties_list:

        for model in models_list:

            fig, axs = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(11, 8))
            fig.subplots_adjust(wspace=0, hspace=0)

            totplots = 0

            for dynamics in dynamics_list:
                # cada net_number
                for net_number in range(TOTAL_NETS):

                    for walk_size in WALK_SIZE_LIST:

                        points = []

                        for walk_number in range(TOTAL_WALKS):

                            for vertex in list(results[model][net_number][walk_size][walk_number][dynamics].keys()):

                                point = results[model][net_number][walk_size][walk_number][dynamics][vertex][prop]

                                points.append(point)


                        x = [x_[0] for x_ in points]
                        y = [y_[1] for y_ in points]

                        #print(x)
                        #print(y)

                        ax0 = totplots // 4
                        ax1 = totplots % 4

                        axs[ax0, ax1].scatter(x, y)

                        #plt.xlabel('Original Network')
                        #plt.ylabel('Reconstructed Network')

                        p_corr, _ = pearsonr(x, y)
                        s_corr, _ = spearmanr(x, y)

                        props_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

                        corr_text = 'pearson = {:.2f}\nspearman = {:.2f}'.format(p_corr, s_corr)
                        axs[ax0, ax1].text(0.05, 0.95, corr_text, transform=axs[ax0, ax1].transAxes,
                                           fontsize=10,
                                           verticalalignment='top', bbox=props_box)

                        #title = models_list[0].upper() + ' k = ' + str(k_list[0]) + ' ' + dynamics.upper() + ' - SIZE = ' + str(walk_size) + ' - property = ' + prop + ' - Pearson = %.3f' % corr


                        #plt.show()

                        totplots += 1

            figname = 'test_' + '_'.join((model.upper(), str(k_list[0]), prop.upper()))

            fig.suptitle(figname.replace('_', ' '))

            fig.text(0.5, 0.04, 'Walk sizes', ha='center', fontsize=10)
            fig.text(0.2, 0.06, '2%', ha='center', fontsize=8)
            fig.text(0.4, 0.06, '10%', ha='center', fontsize=8)
            fig.text(0.6, 0.06, '100%', ha='center', fontsize=8)
            fig.text(0.8, 0.06, '1000%', ha='center', fontsize=8)

            fig.text(0.04, 0.5, 'Dynamics', va='center', rotation='vertical', fontsize=10)
            fig.text(0.08, 0.8, 'RW', va='center', rotation='vertical', fontsize=8)
            fig.text(0.08, 0.6, 'RWD', va='center', rotation='vertical', fontsize=8)
            fig.text(0.08, 0.4, 'RWID', va='center', rotation='vertical', fontsize=8)
            fig.text(0.08, 0.2, 'TSAW', va='center', rotation='vertical', fontsize=8)

            plt.savefig(figname + '.pdf', format='pdf', dpi=100)#, bbox_inches="tight")



def plot_graph1_v2():

    models_list = ['lfr_m05'] #models_list_t + models_list_r
    properties_list = ['degree']# 'closeness']

    WALK_SIZE_LIST = [100, 500, 5000, 50000]

    for prop in properties_list:

        with open('g_wnew_results_' + prop + '.pkl', 'rb') as f:
            results = pickle.load(f)

        for model in models_list:

            fig, axs = plt.subplots(5, 4, sharey=True, figsize=(11, 8))
            fig.subplots_adjust(wspace=0, hspace=0)

            ax0 = -1

            for dynamics in dynamics_list:

                ax0 += 1
                ax1 = -1

                # cada net_number
                for net_number in range(TOTAL_NETS):

                    for walk_size in WALK_SIZE_LIST:

                        ax1 += 1

                        points = []

                        for walk_number in range(TOTAL_WALKS):

                            for vertex in list(results[model][net_number][walk_size][walk_number][dynamics].keys()):

                                point = results[model][net_number][walk_size][walk_number][dynamics][vertex]

                                points.append(point)


                        x = [x_[0] for x_ in points]
                        y = [y_[1] for y_ in points]
                        axs[ax0, ax1].set_xlim([0, min(120, (walk_size / 5000) * 120)])

                        axs[ax0, ax1].scatter(x, y, alpha = 0.1, rasterized=True)

                        p_corr, _ = pearsonr(x, y)
                        s_corr, _ = spearmanr(x, y)

                        #props_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                        props_box = dict(facecolor='white', alpha=0.2)

                        corr_text = '$C_p$ = {:.2f}\n$C_s$ = {:.2f}'.format(p_corr, s_corr)
                        axs[ax0, ax1].text(0.6, 0.9, corr_text, transform=axs[ax0, ax1].transAxes,
                                           fontsize=9,
                                           verticalalignment='top', bbox=props_box)





            figname = 'Points_' + '_'.join((model.upper(), str(k_list[0]), prop.upper()))

            fig.suptitle(figname.replace('_', ' '))

            fig.text(0.5, 0.04, 'Walk sizes', ha='center', fontsize=10)
            fig.text(0.2, 0.90, '2%', ha='center', fontsize=8)
            fig.text(0.4, 0.90, '10%', ha='center', fontsize=8)
            fig.text(0.6, 0.90, '100%', ha='center', fontsize=8)
            fig.text(0.8, 0.90, '1000%', ha='center', fontsize=8)

            fig.text(0.04, 0.5, 'Dynamics', va='center', rotation='vertical', fontsize=10)
            fig.text(0.08, 0.8, 'RW', va='center', rotation='vertical', fontsize=8)
            fig.text(0.08, 0.65, 'RWD', va='center', rotation='vertical', fontsize=8)
            fig.text(0.08, 0.5, 'RWID', va='center', rotation='vertical', fontsize=8)
            fig.text(0.08, 0.35, 'TSAW_Edge', va='center', rotation='vertical', fontsize=8)
            fig.text(0.08, 0.2, 'TSAW_Node', va='center', rotation='vertical', fontsize=8)

            plt.savefig('g_wnew_' + figname + '.pdf', format='pdf', dpi=100)#, bbox_inches="tight")


def plot_graph1_v3():

    models_list = ['lfr_m05'] #models_list_t + models_list_r
    properties_list = ['degree']# 'closeness']

    WALK_SIZE_LIST = [100, 500, 5000, 50000]

    for prop in properties_list:

        with open('g_wnew_results_' + prop + '.pkl', 'rb') as f:
            results = pickle.load(f)

        for model in models_list:

            fig, axs = plt.subplots(5, 4, sharey=True, figsize=(11, 8))
            fig.subplots_adjust(wspace=0, hspace=0)

            ax0 = -1

            for dynamics in dynamics_list:

                ax0 += 1
                ax1 = -1

                # cada net_number
                for net_number in range(TOTAL_NETS):

                    for walk_size in WALK_SIZE_LIST:

                        ax1 += 1

                        points = []

                        for walk_number in range(TOTAL_WALKS):

                            for vertex in list(results[model][net_number][walk_size][walk_number][dynamics].keys()):

                                point = results[model][net_number][walk_size][walk_number][dynamics][vertex]

                                points.append(point)


                        x = [x_[0] for x_ in points]
                        y = [y_[1] for y_ in points]
                        axs[ax0, ax1].set_xlim([0, min(120, (walk_size / 5000) * 120)])

                        axs[ax0, ax1].scatter(x, y, alpha = 0.1, rasterized=True)

                        p_corr, _ = pearsonr(x, y)
                        s_corr, _ = spearmanr(x, y)

                        #props_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                        props_box = dict(facecolor='white', alpha=0.2)

                        corr_text = '$C_p$ = {:.2f}\n$C_s$ = {:.2f}'.format(p_corr, s_corr)
                        axs[ax0, ax1].text(0.6, 0.9, corr_text, transform=axs[ax0, ax1].transAxes,
                                           fontsize=9,
                                           verticalalignment='top', bbox=props_box)





            figname = 'Points_' + '_'.join((model.upper(), str(k_list[0]), prop.upper()))

            #fig.suptitle(figname.replace('_', ' '))

            fig.text(0.5, 0.94, 'Walk size', ha='center', fontsize=10)
            fig.text(0.2, 0.90, '2%', ha='center', fontsize=10)
            fig.text(0.4, 0.90, '10%', ha='center', fontsize=10)
            fig.text(0.6, 0.90, '100%', ha='center', fontsize=10)
            fig.text(0.8, 0.90, '1000%', ha='center', fontsize=10)

            fig.text(0.94, 0.5, 'Dynamics', va='center', rotation=270, fontsize=10)
            fig.text(0.91, 0.8, 'RW', va='center', rotation=270, fontsize=10)
            fig.text(0.91, 0.65, 'RWD', va='center', rotation=270, fontsize=10)
            fig.text(0.91, 0.5, 'RWID', va='center', rotation=270, fontsize=10)
            fig.text(0.91, 0.35, 'TSAW_Edge', va='center', rotation=270, fontsize=10)
            fig.text(0.91, 0.2, 'TSAW_Node', va='center', rotation=270, fontsize=10)

            fig.text(0.06, 0.5, 'Degree (reconstructed network)', va='center', rotation='vertical', fontsize=10)
            fig.text(0.5, 0.04, 'Degree (original network)', ha='center', fontsize=10)


            plt.savefig('ng_wnew_' + figname + '.pdf', format='pdf', dpi=100)#, bbox_inches="tight")



def plot_graph1_v5():

    models_list = ['lfr_m05'] #models_list_t + models_list_r
    properties_list = ['degree']# 'closeness']

    WALK_SIZE_LIST = [100, 500, 5000, 50000]

    for prop in properties_list:

        with open('sg_wnew_results_' + prop + '.pkl', 'rb') as f:
            results = pickle.load(f)

        for model in models_list:

            fig, axs = plt.subplots(5, 4, sharey=True, figsize=(11, 8))
            fig.subplots_adjust(wspace=0, hspace=0)

            ax0 = -1

            for dynamics in dynamics_list:

                ax0 += 1
                ax1 = -1

                # cada net_number
                for net_number in range(TOTAL_NETS):

                    for walk_size in WALK_SIZE_LIST:

                        ax1 += 1

                        points = []

                        for walk_number in range(TOTAL_WALKS):

                            for vertex in list(results[model][net_number][walk_size][walk_number][dynamics].keys()):

                                point = results[model][net_number][walk_size][walk_number][dynamics][vertex]

                                points.append(point)

                        #if dynamics == 'tsaw' and walk_size == 50000:
                        #    print(points)

                        points_reconstructed = [x_[0] for x_ in points]
                        points_original = [y_[1] for y_ in points]

                        print(50*'-')
                        if dynamics == 'tsaw' and walk_size == 50000:
                            print(points_reconstructed)

                        #axs[ax0, ax1].set_xlim([0, min(120, (walk_size / 5000) * 120)])
                        #axs[ax0, ax1].set_xlim([0, max(points_reconstructed)])

                        #axs[ax0, ax1].scatter(points_reconstructed, points_original, alpha = 0.1, rasterized=True)
                        axs[ax0, ax1].scatter(points_reconstructed, points_original, marker = '+', alpha = 0.1, rasterized=True)

                        axs[ax0, ax1].set_ylim([0, max(points_original) * 1.1])
                        axs[ax0, ax1].set_xlim([0, max(points_reconstructed) * 1.5])

                        p_corr, _ = pearsonr(points_reconstructed, points_original)
                        s_corr, _ = spearmanr(points_reconstructed, points_original)

                        #props_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                        props_box = dict(facecolor='white', alpha=0.2)

                        corr_text = '$C_p$ = {:.2f}\n$C_s$ = {:.2f}'.format(p_corr, s_corr)
                        axs[ax0, ax1].text(0.65, 0.3, corr_text, transform=axs[ax0, ax1].transAxes,
                                           fontsize=9,
                                           verticalalignment='top', bbox=props_box)

            figname = 'Points_' + '_'.join((model.upper(), str(k_list[0]), prop.upper()))

            #fig.suptitle(figname.replace('_', ' '))

            fig.text(0.5, 0.94, 'Walk size', ha='center', fontsize=10)
            fig.text(0.2, 0.90, '100', ha='center', fontsize=10)
            fig.text(0.4, 0.90, '500', ha='center', fontsize=10)
            fig.text(0.6, 0.90, '5000', ha='center', fontsize=10)
            fig.text(0.8, 0.90, '50000', ha='center', fontsize=10)

            fig.text(0.94, 0.5, 'Dynamics', va='center', rotation=270, fontsize=10)
            fig.text(0.91, 0.8, 'RW', va='center', rotation=270, fontsize=10)
            fig.text(0.91, 0.65, 'RWD', va='center', rotation=270, fontsize=10)
            fig.text(0.91, 0.5, 'RWID', va='center', rotation=270, fontsize=10)
            fig.text(0.91, 0.35, 'TSAW_Edge', va='center', rotation=270, fontsize=10)
            fig.text(0.91, 0.2, 'TSAW_Node', va='center', rotation=270, fontsize=10)

            fig.text(0.06, 0.5, 'Degree (original network)', va='center', rotation='vertical', fontsize=10)
            fig.text(0.5, 0.04, 'Degree (reconstructed network)', ha='center', fontsize=10)


            plt.savefig('sg_wnew5_' + figname + '.pdf', format='pdf', dpi=100)#, bbox_inches="tight")



def plot_graph1_v4():

    models_list = ['lfr_m05'] #models_list_t + models_list_r
    properties_list = ['degree']# 'closeness']

    WALK_SIZE_LIST = [100, 500, 5000, 50000]

    for prop in properties_list:

        with open('sg_wnew_results_' + prop + '.pkl', 'rb') as f:
            results = pickle.load(f)

        for model in models_list:

            fig, axs = plt.subplots(5, 4, sharey=True, figsize=(11, 8))
            fig.subplots_adjust(wspace=0, hspace=0)

            ax0 = -1

            for dynamics in dynamics_list:

                ax0 += 1
                ax1 = -1

                # cada net_number
                for net_number in range(TOTAL_NETS):

                    for walk_size in WALK_SIZE_LIST:

                        ax1 += 1

                        points = []

                        for walk_number in range(TOTAL_WALKS):

                            for vertex in list(results[model][net_number][walk_size][walk_number][dynamics].keys()):

                                point = results[model][net_number][walk_size][walk_number][dynamics][vertex]

                                points.append(point)

                        #if dynamics == 'tsaw' and walk_size == 50000:
                        #    print(points)

                        points_reconstructed = [x_[0] for x_ in points]
                        points_original = [y_[1] for y_ in points]

                        print(50*'-')
                        if dynamics == 'tsaw' and walk_size == 50000:
                            print(points_reconstructed)

                        #axs[ax0, ax1].set_xlim([0, min(120, (walk_size / 5000) * 120)])
                        #axs[ax0, ax1].set_xlim([0, max(points_reconstructed)])

                        #axs[ax0, ax1].scatter(points_reconstructed, points_original, alpha = 0.1, rasterized=True)
                        axs[ax0, ax1].plot(points_original, points_reconstructed, marker = 'o', alpha = 0.1, rasterized=True)

                        axs[ax0, ax1].set_xlim([0, max(points_original) * 1.2])
                        axs[ax0, ax1].set_ylim([0, max(points_reconstructed) * 2.0])

                        p_corr, _ = pearsonr(points_reconstructed, points_original)
                        s_corr, _ = spearmanr(points_reconstructed, points_original)

                        #props_box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                        props_box = dict(facecolor='white', alpha=0.2)

                        corr_text = '$C_p$ = {:.2f}\n$C_s$ = {:.2f}'.format(p_corr, s_corr)
                        axs[ax0, ax1].text(0.6, 0.9, corr_text, transform=axs[ax0, ax1].transAxes,
                                           fontsize=9,
                                           verticalalignment='top', bbox=props_box)

            figname = 'Points_' + '_'.join((model.upper(), str(k_list[0]), prop.upper()))

            #fig.suptitle(figname.replace('_', ' '))

            fig.text(0.5, 0.94, 'Walk size', ha='center', fontsize=10)
            fig.text(0.2, 0.90, '100', ha='center', fontsize=10)
            fig.text(0.4, 0.90, '500', ha='center', fontsize=10)
            fig.text(0.6, 0.90, '5000', ha='center', fontsize=10)
            fig.text(0.8, 0.90, '50000', ha='center', fontsize=10)

            fig.text(0.94, 0.5, 'Dynamics', va='center', rotation=270, fontsize=10)
            fig.text(0.91, 0.8, 'RW', va='center', rotation=270, fontsize=10)
            fig.text(0.91, 0.65, 'RWD', va='center', rotation=270, fontsize=10)
            fig.text(0.91, 0.5, 'RWID', va='center', rotation=270, fontsize=10)
            fig.text(0.91, 0.35, 'TSAW_Edge', va='center', rotation=270, fontsize=10)
            fig.text(0.91, 0.2, 'TSAW_Node', va='center', rotation=270, fontsize=10)

            fig.text(0.06, 0.5, 'Degree (reconstructed network)', va='center', rotation='vertical', fontsize=10)
            fig.text(0.5, 0.04, 'Degree (original network)', ha='center', fontsize=10)


            plt.savefig('sg_wnew_' + figname + '.pdf', format='pdf', dpi=100)#, bbox_inches="tight")



def plot_correlation_curve(metric='pearson'):
    #pearson or spearman

    for prop in properties_list:

        with open('g_wnew_results_' + prop + '.pkl', 'rb') as f:
            results = pickle.load(f)

        for model in models_list_t:

            result_plot = {}

            for net_number in range(TOTAL_NETS):

                for walk_size in WALK_SIZE_LIST:

                    result_plot[walk_size] = {}

                    for dynamics in dynamics_list:

                        points = []

                        for walk_number in range(TOTAL_WALKS):

                            for vertex in list(results[model][net_number][walk_size][walk_number][dynamics].keys()):
                                point = results[model][net_number][walk_size][walk_number][dynamics][vertex]

                                points.append(point)

                        x = [x_[0] if math.isnan(float(x_[0])) is False else 0 for x_ in points]
                        y = [y_[1] if math.isnan(float(y_[1])) is False else 0 for y_ in points]

                        print(model, dynamics)
                        print('x->', x)
                        print('y->', y)

                        p_corr, _ = pearsonr(x, y)
                        s_corr, _ = spearmanr(x, y)

                        print(p_corr)

                        result_plot[walk_size][dynamics] = {}

                        result_plot[walk_size][dynamics]['pearson'] = p_corr
                        result_plot[walk_size][dynamics]['spearman'] = s_corr

            y_rw = []
            y_rwd = []
            y_rwid = []
            y_tsaw = []
            y_tsaw_node = []

            fig1, ax1 = plt.subplots()

            for w in WALK_SIZE_LIST:
                y_rw.append(result_plot[w]['rw'][metric])
                y_rwd.append(result_plot[w]['rwd'][metric])
                y_rwid.append(result_plot[w]['rwid'][metric])
                y_tsaw.append(result_plot[w]['tsaw'][metric])
                y_tsaw_node.append(result_plot[w]['tsaw_node'][metric])

            ax1.plot(WALK_SIZE_LIST, y_rw, marker='o', label='RW', color='red')
            ax1.plot(WALK_SIZE_LIST, y_rwd, marker='s', label='RWD', color='green')
            ax1.plot(WALK_SIZE_LIST, y_rwid, marker='v', label='RWID', color='blue')
            ax1.plot(WALK_SIZE_LIST, y_tsaw, marker='D', label='TSAW_Edge', color='orange')
            ax1.plot(WALK_SIZE_LIST, y_tsaw_node, marker='+', label='TSAW_Node', color='lightblue')

            ax1.set_xscale('log')
            ax1.set_xticks(WALK_SIZE_LIST)
            #ax1.set_xticklabels(WALK_SIZE_LIST, rotation=45, ha='right')
            ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


            plt.title(str(model).upper() + ' - ' + str(prop).upper() + ' - ' + str(metric).upper())
            plt.legend()
            plt.tight_layout()

            figname = '_'.join(('g_wnew', str(model), str(prop), str(metric)))

            plt.savefig(figname + '.pdf', format='pdf', dpi=500)

            plt.clf()


def plot_metric_vs_knowledge(metric='pearson'):

    #plotar curva de pearson e conhecimento
    #eixo y eh igual plot_correlation_curve, eixo x é o numero de nós descobertos
    #fazer para todos os modelos

    properties_list = ['knowledge']

    for prop in properties_list:

        for model in models_list:

            result_plot = {}

            for net_number in range(TOTAL_NETS):

                for walk_size in WALK_SIZE_LIST:


                    print(walk_size)

                    result_plot[walk_size] = {}

                    for dynamics in dynamics_list:

                        points = []

                        for walk_number in range(TOTAL_WALKS):

                            for vertex in list(results[model][net_number][walk_size][walk_number][dynamics].keys()):

                                point = results[model][net_number][walk_size][walk_number][dynamics][vertex][prop]

                                points.append(point)



                        x_corr = [x_[1][0] for x_ in points]
                        y_corr = [y_[1][1] for y_ in points]

                        p_corr, _ = pearsonr(x_corr, y_corr)
                        s_corr, _ = spearmanr(x_corr, y_corr)

                        x = points[-1][0][0]

                        #print('tamanhos', len(x_corr), len(y_corr), len(x))

                        result_plot[walk_size][dynamics] = {}

                        result_plot[walk_size][dynamics]['pearson'] = p_corr
                        result_plot[walk_size][dynamics]['spearman'] = s_corr

                        result_plot[walk_size][dynamics]['x'] = x




            y_rw = []
            y_rwd = []
            y_rwid = []
            y_tsaw = []
            y_tsaw_node = []

            x_rw = []
            x_rwd = []
            x_rwid = []
            x_tsaw = []
            x_tsaw_node = []

            fig1, ax1 = plt.subplots()

            for w in WALK_SIZE_LIST:
                y_rw.append(result_plot[w]['rw'][metric])
                y_rwd.append(result_plot[w]['rwd'][metric])
                y_rwid.append(result_plot[w]['rwid'][metric])
                y_tsaw.append(result_plot[w]['tsaw'][metric])
                y_tsaw_node.append(result_plot[w]['tsaw_node'][metric])

                x_rw.append(result_plot[w]['rw']['x'])
                x_rwd.append(result_plot[w]['rwd']['x'])
                x_rwid.append(result_plot[w]['rwid']['x'])
                x_tsaw.append(result_plot[w]['tsaw']['x'])
                x_tsaw_node.append(result_plot[w]['tsaw_node']['x'])

                #for d in dynamics_list:
                #plt.plot(w, result_plot[w]['rw']['pearson'], label='RW', color='red')
                #plt.plot(w, result_plot[w]['rwd']['pearson'], label='RWD', color='green')
                #plt.plot(w, result_plot[w]['rwid']['pearson'], label='RWID', color='blue')
                #plt.plot(w, result_plot[w]['tsaw']['pearson'], label='TSAW', color='orange')
                #plt.plot(w, result_plot[w]['tsaw_node']['pearson'], label='TSAW_NODE', color='lightblue')

            print('x', x_rw)
            print('y', y_rw)

            ax1.plot(x_rw, y_rw, marker='o', label='RW', color='red')
            ax1.plot(x_rwd, y_rwd, marker='s', label='RWD', color='green')
            ax1.plot(x_rwid, y_rwid, marker='v', label='RWID', color='blue')
            ax1.plot(x_tsaw, y_tsaw, marker='D', label='TSAW', color='orange')
            ax1.plot(x_tsaw_node, y_tsaw_node, marker='+', label='TSAW_NODE', color='lightblue')

            #ax1.set_xscale('log')
            #ax1.set_xticks(WALK_SIZE_LIST)

            #ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


            plt.title('knowledge_' + str(model).upper() + ' - ' + str(prop).upper() + ' - ' + str(metric).upper())
            plt.legend()
            plt.tight_layout()

            figname = 'knowledge_correlation_' + '_'.join((str(model), str(prop), str(metric)))

            plt.savefig(figname + '.pdf', format='pdf', dpi=500)

            plt.clf()






def plot_metric_vs_knowledge_average(metric='pearson'):

    #plotar curva de pearson e conhecimento
    #eixo y eh igual plot_correlation_curve, eixo x é o numero de nós descobertos
    #fazer para todos os modelos


    # propriedade a fazer combinação com as demais
    prop1 = 'knowledge'

    #propriedades a traçar vs prop1
    #properties_list = ['cc']
    #models_list = ['lfr_m05']
    #models_list = ['er', 'ba', 'wax', 'lfr', 'facebook', 'power']

    with open('results_' + prop1 + '.pkl', 'rb') as f:
        results_knowledge = pickle.load(f)

    print('aqui')

    for prop in properties_list:

        print('iniciando')

        with open('results_' + prop + '.pkl', 'rb') as f:
            results_prop = pickle.load(f)

        for model in models_list:

            result_plot = {}

            for net_number in range(TOTAL_NETS):

                for walk_size in WALK_SIZE_LIST:

                    result_plot[walk_size] = {}

                    for dynamics in dynamics_list:

                        pointsx = []
                        pointsy = []

                        list_avg_pointsx = []

                        for walk_number in range(TOTAL_WALKS):

                            for vertex in list(results_prop[model][net_number][walk_size][walk_number][dynamics].keys()):

                                pointx = results_knowledge[model][net_number][walk_size][walk_number][dynamics][vertex]
                                pointy = results_prop[model][net_number][walk_size][walk_number][dynamics][vertex]


                                pointsx.append(pointx)
                                pointsy.append(pointy)

                            list_avg_pointsx.append(pointsx[-1][0])

                        avg_point = sum(list_avg_pointsx)/len(list_avg_pointsx)

                        x_corr = [x_[0] for x_ in pointsy]
                        y_corr = [y_[1] for y_ in pointsy]

                        x_corr = [x_ if math.isnan(float(x_)) is False else 0 for x_ in x_corr]
                        y_corr = [y_ if math.isnan(float(y_)) is False else 0 for y_ in y_corr]

                        print(x_corr)
                        print(50 * '---')
                        print(y_corr)

                        p_corr, _ = pearsonr(x_corr, y_corr)
                        s_corr, _ = spearmanr(x_corr, y_corr)

                        #print('tamanhos', len(x_corr), len(y_corr), len(x))

                        result_plot[walk_size][dynamics] = {}

                        result_plot[walk_size][dynamics]['pearson'] = p_corr
                        result_plot[walk_size][dynamics]['spearman'] = s_corr

                        result_plot[walk_size][dynamics]['x'] = avg_point





            y_rw = []
            y_rwd = []
            y_rwid = []
            y_tsaw = []
            y_tsaw_node = []

            x_rw = []
            x_rwd = []
            x_rwid = []
            x_tsaw = []
            x_tsaw_node = []

            fig1, ax1 = plt.subplots()

            for w in WALK_SIZE_LIST:
                y_rw.append(result_plot[w]['rw'][metric])
                y_rwd.append(result_plot[w]['rwd'][metric])
                y_rwid.append(result_plot[w]['rwid'][metric])
                y_tsaw.append(result_plot[w]['tsaw'][metric])
                y_tsaw_node.append(result_plot[w]['tsaw_node'][metric])

                x_rw.append(result_plot[w]['rw']['x'])
                x_rwd.append(result_plot[w]['rwd']['x'])
                x_rwid.append(result_plot[w]['rwid']['x'])
                x_tsaw.append(result_plot[w]['tsaw']['x'])
                x_tsaw_node.append(result_plot[w]['tsaw_node']['x'])

                #for d in dynamics_list:
                #plt.plot(w, result_plot[w]['rw']['pearson'], label='RW', color='red')
                #plt.plot(w, result_plot[w]['rwd']['pearson'], label='RWD', color='green')
                #plt.plot(w, result_plot[w]['rwid']['pearson'], label='RWID', color='blue')
                #plt.plot(w, result_plot[w]['tsaw']['pearson'], label='TSAW', color='orange')
                #plt.plot(w, result_plot[w]['tsaw_node']['pearson'], label='TSAW_NODE', color='lightblue')


            ax1.plot(x_rw, y_rw, marker='o', label='RW', color='red')
            ax1.plot(x_rwd, y_rwd, marker='s', label='RWD', color='green')
            ax1.plot(x_rwid, y_rwid, marker='v', label='RWID', color='blue')
            ax1.plot(x_tsaw, y_tsaw, marker='D', label='TSAW', color='orange')
            ax1.plot(x_tsaw_node, y_tsaw_node, marker='+', label='TSAW_NODE', color='lightblue')

            #ax1.set_xscale('log')
            #ax1.set_xticks(WALK_SIZE_LIST)

            #ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())


            plt.title('knowledge - ' + str(model).upper() + ' - ' + str(prop).upper() + ' - ' + str(metric).upper())
            plt.legend()
            plt.tight_layout()

            figname = 'r_avg_knowledge_correlation_' + '_'.join((str(model), str(prop), str(metric)))

            plt.savefig(figname + '.pdf', format='pdf', dpi=500)

            plt.clf()

            print('finished', model, prop)


def plot_community_nmi():

    prop = 'community_ml'

    with open('results_' + prop + '.pkl', 'rb') as f:
        results = pickle.load(f)

    for model in models_list:

        result_plot = {}

        for net_number in range(TOTAL_NETS):

            for walk_size in WALK_SIZE_LIST:

                print(walk_size)

                result_plot[walk_size] = {}

                for dynamics in dynamics_list:

                    nmis = []

                    for walk_number in range(TOTAL_WALKS):

                        points = []

                        for vertex in list(results[model][net_number][walk_size][walk_number][dynamics].keys()):
                            point = results[model][net_number][walk_size][walk_number][dynamics][vertex]

                            points.append(point)

                        points_reconstructed = [p[0] for p in points]
                        points_original = [p[1] for p in points]

                        nmi = metrics.normalized_mutual_info_score(points_reconstructed, points_original)

                        nmis.append(nmi)

                    avg_nmi = sum(nmis) / len(nmis)
                    result_plot[walk_size][dynamics] = avg_nmi

        y_rw = []
        y_rwd = []
        y_rwid = []
        y_tsaw = []
        y_tsaw_node = []

        fig1, ax1 = plt.subplots()

        for w in WALK_SIZE_LIST:
            y_rw.append(result_plot[w]['rw'])
            y_rwd.append(result_plot[w]['rwd'])
            y_rwid.append(result_plot[w]['rwid'])
            y_tsaw.append(result_plot[w]['tsaw'])
            y_tsaw_node.append(result_plot[w]['tsaw_node'])


        ax1.plot(WALK_SIZE_LIST, y_rw, marker='o', label='RW', color='red')
        ax1.plot(WALK_SIZE_LIST, y_rwd, marker='s', label='RWD', color='green')
        ax1.plot(WALK_SIZE_LIST, y_rwid, marker='v', label='RWID', color='blue')
        ax1.plot(WALK_SIZE_LIST, y_tsaw, marker='D', label='TSAW', color='orange')
        ax1.plot(WALK_SIZE_LIST, y_tsaw_node, marker='+', label='TSAW_NODE', color='lightblue')

        ax1.set_xscale('log')
        ax1.set_xticks(WALK_SIZE_LIST)
        # ax1.set_xticklabels(WALK_SIZE_LIST, rotation=45, ha='right')
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        plt.title(str(model).upper() + ' - ' + str(prop).upper())# + ' - ' + str(metric).upper())
        plt.legend()
        plt.tight_layout()

        figname = '_'.join(('r', str(model), str('communities - multilevel')))

        plt.savefig(figname + '.pdf', format='pdf', dpi=500)

        plt.clf()


def group_correlation(theorical_real = 't', metric='pearson'):

    if theorical_real == 't':
        models_list = models_list_t
        models_list_plot = models_list_plot_t
    elif theorical_real == 'r':
        models_list = models_list_r
        models_list_plot = models_list_plot_r

    fig, axs = plt.subplots(len(models_list), len(properties_list), sharex=True, sharey=True, figsize=(11, 8))
    fig.subplots_adjust(wspace=0, hspace=0)

    totplots = 0

    ax1 = 0

    for prop in properties_list:

        ax0 = 0

        with open('sg_wnew_results_' + prop + '.pkl', 'rb') as f:
            results = pickle.load(f)

        for model in models_list:

            print(prop, model)

            result_plot = {}

            for net_number in range(TOTAL_NETS):

                for walk_size in WALK_SIZE_LIST:

                    result_plot[walk_size] = {}

                    for dynamics in dynamics_list:

                        points = []

                        for walk_number in range(TOTAL_WALKS):

                            for vertex in list(results[model][net_number][walk_size][walk_number][dynamics].keys()):
                                point = results[model][net_number][walk_size][walk_number][dynamics][vertex]

                                points.append(point)

                        x = [x_[0] if math.isnan(float(x_[0])) is False else 0 for x_ in points]
                        y = [y_[1] if math.isnan(float(y_[1])) is False else 0 for y_ in points]

                        p_corr, _ = pearsonr(x, y)
                        s_corr, _ = spearmanr(x, y)

                        result_plot[walk_size][dynamics] = {}

                        result_plot[walk_size][dynamics]['pearson'] = p_corr
                        result_plot[walk_size][dynamics]['spearman'] = s_corr

            y_rw = []
            y_rwd = []
            y_rwid = []
            y_tsaw = []
            y_tsaw_node = []

            for w in WALK_SIZE_LIST:
                y_rw.append(result_plot[w]['rw'][metric])
                y_rwd.append(result_plot[w]['rwd'][metric])
                y_rwid.append(result_plot[w]['rwid'][metric])
                y_tsaw.append(result_plot[w]['tsaw'][metric])
                y_tsaw_node.append(result_plot[w]['tsaw_node'][metric])

            axs[ax0, ax1].axhline(y=0.0, color='lightgray', linestyle='--')
            axs[ax0, ax1].axhline(y=0.5, color='lightgray', linestyle='--')
            axs[ax0, ax1].axhline(y=1.0, color='lightgray', linestyle='--')

            axs[ax0, ax1].plot(WALK_SIZE_LIST, y_rw, ms=3, marker='o', label='RW', color='red')
            axs[ax0, ax1].plot(WALK_SIZE_LIST, y_rwd, ms=3, marker='s', label='RWD', color='green')
            axs[ax0, ax1].plot(WALK_SIZE_LIST, y_rwid, ms=3, marker='v', label='RWID', color='blue')
            axs[ax0, ax1].plot(WALK_SIZE_LIST, y_tsaw, ms=3, marker='D', label='TSAW_Edge', color='orange')
            axs[ax0, ax1].plot(WALK_SIZE_LIST, y_tsaw_node, ms=3, marker='+', label='TSAW_Node', color='lightblue')

            axs[ax0, ax1].set_xscale('log')
            axs[ax0, ax1].set_xticks(WALK_SIZE_LIST)


            #
            axs[ax0, ax1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

            axs[ax0, ax1].set_xticklabels(WALK_SIZE_LIST, rotation=45, fontsize=6, ha='right')


            #plt.title(str(model).upper() + ' - ' + str(prop).upper() + ' - ' + str(metric).upper())
            #plt.legend()
            #plt.tight_layout()

            #figname = '_'.join(('r', str(model), str(prop), str(metric)))

            #plt.savefig(figname + '.pdf', format='pdf', dpi=500)

            #plt.clf()

            totplots += 1

            ax0 += 1

        ax1 += 1

    print('finish')



    # eixo x
    fig.text(0.5, 0.01, 'PROPERTIES', ha='center', fontsize=11)
    fig.text(0.17, 0.04, properties_list_plot[0], ha='center', fontsize=9)
    fig.text(0.31, 0.04, properties_list_plot[1], ha='center', fontsize=9)
    fig.text(0.45, 0.04, properties_list_plot[2], ha='center', fontsize=9)
    fig.text(0.57, 0.04, properties_list_plot[3], ha='center', fontsize=9)
    fig.text(0.7, 0.04, properties_list_plot[4], ha='center', fontsize=9)
    fig.text(0.84, 0.04, properties_list_plot[5], ha='center', fontsize=9)

    # eixo y real
    if theorical_real == 'r':
        fig.text(0.04, 0.5, 'MODELS', va='center', rotation='vertical', fontsize=11)
        fig.text(0.08, 0.85, models_list_plot[0], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.71, models_list_plot[1], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.60, models_list_plot[2], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.48, models_list_plot[3], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.37, models_list_plot[4], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.26, models_list_plot[5], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.15, models_list_plot[6], va='center', rotation='vertical', fontsize=9)

    # eixo y models
    if theorical_real == 't':
        fig.text(0.04, 0.5, 'MODELS', va='center', rotation='vertical', fontsize=11)
        fig.text(0.08, 0.8, models_list_plot[0], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.68, models_list_plot[1], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.55, models_list_plot[2], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.44, models_list_plot[3], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.32, models_list_plot[4], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.2, models_list_plot[5], va='center', rotation='vertical', fontsize=9)

    #fig.legend(loc='center right', borderaxespad=0.1)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    long_type = 'models' if theorical_real == 't' else 'real'

    plt.savefig('sg_wnew_' + long_type + '_all_' + metric + '.pdf', format='pdf', dpi=100)  # , bbox_inches="tight")




def group_knowledge_correlation(theorical_real = 't', metric='pearson'):

    if theorical_real == 't':
        models_list = models_list_t
        models_list_plot = models_list_plot_t
    elif theorical_real == 'r':
        models_list = models_list_r
        models_list_plot = models_list_plot_r

    """
    network_size = {
        'facebook': 320,
        'power': 4941,
        'econ-poli': 3915,
        'web-EPA': 4271,
        'Collins': 1622,
        'AI_interactions': 4829,
        'socfb-JohnsHopkins55': 5180,
        'er': 5000,
        'ba': 5000,
        'wax': 5000,
        'lfr': 5000,
        'lfr_m05': 5000,
        'lfr_m8': 5000
    }
    """

    fig, axs = plt.subplots(len(models_list), len(properties_list), sharex=True, sharey=True, figsize=(11, 8))
    fig.subplots_adjust(wspace=0, hspace=0)

    totplots = 0

    ax1 = 0

    prop1 = 'knowledge'

    with open('results_' + prop1 + '.pkl', 'rb') as f:
        results_knowledge = pickle.load(f)

    for prop in properties_list:

        ax0 = 0

        with open('results_' + prop + '.pkl', 'rb') as f:
            results_prop = pickle.load(f)

        for model in models_list:

            print(prop, model)

            result_plot = {}

            for net_number in range(TOTAL_NETS):

                for walk_size in WALK_SIZE_LIST:

                    result_plot[walk_size] = {}

                    for dynamics in dynamics_list:

                        pointsx = []
                        pointsy = []

                        list_avg_pointsx = []

                        for walk_number in range(TOTAL_WALKS):

                            for vertex in list(results_prop[model][net_number][walk_size][walk_number][dynamics].keys()):
                                pointx = results_knowledge[model][net_number][walk_size][walk_number][dynamics][vertex]
                                pointy = results_prop[model][net_number][walk_size][walk_number][dynamics][vertex]

                                pointsx.append(pointx)
                                pointsy.append(pointy)

                            list_avg_pointsx.append(pointsx[-1][0])

                        avg_point = sum(list_avg_pointsx) / len(list_avg_pointsx)

                        x_corr = [x_[0] for x_ in pointsy]
                        y_corr = [y_[1] for y_ in pointsy]

                        x_corr = [x_ if math.isnan(float(x_)) is False else 0 for x_ in x_corr]
                        y_corr = [y_ if math.isnan(float(y_)) is False else 0 for y_ in y_corr]


                        p_corr, _ = pearsonr(x_corr, y_corr)
                        s_corr, _ = spearmanr(x_corr, y_corr)

                        result_plot[walk_size][dynamics] = {}

                        result_plot[walk_size][dynamics]['pearson'] = p_corr
                        result_plot[walk_size][dynamics]['spearman'] = s_corr

                        result_plot[walk_size][dynamics]['x'] = avg_point

            y_rw = []
            y_rwd = []
            y_rwid = []
            y_tsaw = []
            y_tsaw_node = []

            x_rw = []
            x_rwd = []
            x_rwid = []
            x_tsaw = []
            x_tsaw_node = []

            for w in WALK_SIZE_LIST:
                y_rw.append(result_plot[w]['rw'][metric])
                y_rwd.append(result_plot[w]['rwd'][metric])
                y_rwid.append(result_plot[w]['rwid'][metric])
                y_tsaw.append(result_plot[w]['tsaw'][metric])
                y_tsaw_node.append(result_plot[w]['tsaw_node'][metric])

                x_rw.append(result_plot[w]['rw']['x'])
                x_rwd.append(result_plot[w]['rwd']['x'])
                x_rwid.append(result_plot[w]['rwid']['x'])
                x_tsaw.append(result_plot[w]['tsaw']['x'])
                x_tsaw_node.append(result_plot[w]['tsaw_node']['x'])

            axs[ax0, ax1].plot(x_rw, y_rw, ms=3, marker='o', label='RW', color='red')
            axs[ax0, ax1].plot(x_rwd, y_rwd, ms=3, marker='s', label='RWD', color='green')
            axs[ax0, ax1].plot(x_rwid, y_rwid, ms=3, marker='v', label='RWID', color='blue')
            axs[ax0, ax1].plot(x_tsaw, y_tsaw, ms=3, marker='D', label='TSAW', color='orange')
            axs[ax0, ax1].plot(x_tsaw_node, y_tsaw_node, ms=3, marker='+', label='TSAW_NODE', color='lightblue')

            #axs[ax0, ax1].set_xscale('log')
            #axs[ax0, ax1].set_xticks(WALK_SIZE_LIST)
            #
            axs[ax0, ax1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axs[ax0, ax1].set_xlim([-500, 500 + max(x_rw[-1], x_rwd[-1], x_rwid[-1], x_tsaw[-1], x_tsaw_node[-1])])


            #axs[ax0, ax1].set_xticklabels(rotation=45, fontsize=6, ha='right')

            totplots += 1

            ax0 += 1

        ax1 += 1

    print('finish')



    # eixo x
    fig.text(0.5, 0.01, 'PROPERTIES', ha='center', fontsize=11)
    fig.text(0.2, 0.04, properties_list_plot[0], ha='center', fontsize=9)
    fig.text(0.35, 0.04, properties_list_plot[1], ha='center', fontsize=9)
    fig.text(0.5, 0.04, properties_list_plot[2], ha='center', fontsize=9)
    fig.text(0.65, 0.04, properties_list_plot[3], ha='center', fontsize=9)
    fig.text(0.8, 0.04, properties_list_plot[4], ha='center', fontsize=9)

    # eixo y real
    if theorical_real == 'r':
        fig.text(0.04, 0.5, 'MODELS', va='center', rotation='vertical', fontsize=11)
        fig.text(0.08, 0.85, models_list_plot[0], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.71, models_list_plot[1], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.60, models_list_plot[2], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.48, models_list_plot[3], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.37, models_list_plot[4], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.26, models_list_plot[5], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.15, models_list_plot[6], va='center', rotation='vertical', fontsize=9)

    # eixo y models
    if theorical_real == 't':
        fig.text(0.04, 0.5, 'MODELS', va='center', rotation='vertical', fontsize=11)
        fig.text(0.08, 0.8, models_list_plot[0], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.68, models_list_plot[1], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.55, models_list_plot[2], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.44, models_list_plot[3], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.32, models_list_plot[4], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.2, models_list_plot[5], va='center', rotation='vertical', fontsize=9)

    #fig.legend(loc='center right', borderaxespad=0.1)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    long_type = 'models' if theorical_real == 't' else 'real'
    plt.savefig('knowledge_all_' + long_type + '_' + metric + '.pdf', format='pdf', dpi=100)  # , bbox_inches="tight")



def group_knowledge_correlation_perc(theorical_real = 't', metric='pearson'):



    if theorical_real == 't':
        models_list = models_list_t
        models_list_plot = models_list_plot_t
    elif theorical_real == 'r':
        models_list = models_list_r
        models_list_plot = models_list_plot_r

    """
    network_size = {
        'facebook': 320,
        'power': 4941,
        'econ-poli': 3915,
        'web-EPA': 4271,
        'Collins': 1622,
        'AI_interactions': 4829,
        'socfb-JohnsHopkins55': 5180,
        'er': 5000,
        'ba': 5000,
        'wax': 5000,
        'lfr': 5000,
        'lfr_m05': 5000,
        'lfr_m8': 5000
    }
    """

    network_size = {
        'facebook': 320,
        'power': 4941,
        'econ-poli': 2343,
        'web-EPA': 4253,
        'Collins': 1004,
        'bio-DM-CX': 4032,
        'AI_interactions': 4519,
        'socfb-JohnsHopkins55': 5157,
        'er': 5001,
        'ba': 5000,
        'wax': 4900,
        'lfr': 5000,
        'lfr_m05': 5000,
        'lfr_m8': 5000
    }

    fig, axs = plt.subplots(len(models_list), len(properties_list), sharex=True, sharey=True, figsize=(11, 8))
    fig.subplots_adjust(wspace=0, hspace=0)

    totplots = 0

    ax1 = 0

    prop1 = 'knowledge'

    with open('g_wnew_results_' + prop1 + '.pkl', 'rb') as f:
        results_knowledge = pickle.load(f)

    for prop in properties_list:

        ax0 = 0

        with open('g_wnew_results_' + prop + '.pkl', 'rb') as f:
            results_prop = pickle.load(f)

        for model in models_list:

            print(prop, model)

            result_plot = {}

            for net_number in range(TOTAL_NETS):

                for walk_size in WALK_SIZE_LIST:

                    result_plot[walk_size] = {}

                    for dynamics in dynamics_list:

                        pointsx = []
                        pointsy = []

                        list_avg_pointsx = []

                        for walk_number in range(TOTAL_WALKS):

                            for vertex in list(results_prop[model][net_number][walk_size][walk_number][dynamics].keys()):
                                pointx = results_knowledge[model][net_number][walk_size][walk_number][dynamics][vertex]
                                pointy = results_prop[model][net_number][walk_size][walk_number][dynamics][vertex]

                                pointsx.append(pointx)
                                pointsy.append(pointy)

                            list_avg_pointsx.append(pointsx[-1][0])

                        avg_point = sum(list_avg_pointsx) / len(list_avg_pointsx)
                        avg_point = avg_point / network_size[model]

                        x_corr = [x_[0] for x_ in pointsy]
                        y_corr = [y_[1] for y_ in pointsy]

                        x_corr = [x_ if math.isnan(float(x_)) is False else 0 for x_ in x_corr]
                        y_corr = [y_ if math.isnan(float(y_)) is False else 0 for y_ in y_corr]


                        p_corr, _ = pearsonr(x_corr, y_corr)
                        s_corr, _ = spearmanr(x_corr, y_corr)

                        result_plot[walk_size][dynamics] = {}

                        result_plot[walk_size][dynamics]['pearson'] = p_corr
                        result_plot[walk_size][dynamics]['spearman'] = s_corr

                        result_plot[walk_size][dynamics]['x'] = avg_point

            y_rw = []
            y_rwd = []
            y_rwid = []
            y_tsaw = []
            y_tsaw_node = []

            x_rw = []
            x_rwd = []
            x_rwid = []
            x_tsaw = []
            x_tsaw_node = []

            for w in WALK_SIZE_LIST:
                y_rw.append(result_plot[w]['rw'][metric])
                y_rwd.append(result_plot[w]['rwd'][metric])
                y_rwid.append(result_plot[w]['rwid'][metric])
                y_tsaw.append(result_plot[w]['tsaw'][metric])
                y_tsaw_node.append(result_plot[w]['tsaw_node'][metric])

                x_rw.append(result_plot[w]['rw']['x'])
                x_rwd.append(result_plot[w]['rwd']['x'])
                x_rwid.append(result_plot[w]['rwid']['x'])
                x_tsaw.append(result_plot[w]['tsaw']['x'])
                x_tsaw_node.append(result_plot[w]['tsaw_node']['x'])

            axs[ax0, ax1].plot(x_rw, y_rw, ms=3, marker='o', label='RW', color='red')
            axs[ax0, ax1].plot(x_rwd, y_rwd, ms=3, marker='s', label='RWD', color='green')
            axs[ax0, ax1].plot(x_rwid, y_rwid, ms=3, marker='v', label='RWID', color='blue')
            axs[ax0, ax1].plot(x_tsaw, y_tsaw, ms=3, marker='D', label='TSAW_Edge', color='orange')
            axs[ax0, ax1].plot(x_tsaw_node, y_tsaw_node, ms=3, marker='+', label='TSAW_Node', color='lightblue')

            #axs[ax0, ax1].set_xscale('log')
            #axs[ax0, ax1].set_xticks(WALK_SIZE_LIST)
            #
            axs[ax0, ax1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axs[ax0, ax1].set_xlim([-0.05, 1.05])


            #axs[ax0, ax1].set_xticklabels(rotation=45, fontsize=6, ha='right')

            totplots += 1

            ax0 += 1

        ax1 += 1

    print('finish')



    # eixo x
    """
    fig.text(0.5, 0.01, 'PROPERTIES', ha='center', fontsize=11)
    fig.text(0.2, 0.04, properties_list_plot[0], ha='center', fontsize=9)
    fig.text(0.35, 0.04, properties_list_plot[1], ha='center', fontsize=9)
    fig.text(0.5, 0.04, properties_list_plot[2], ha='center', fontsize=9)
    fig.text(0.65, 0.04, properties_list_plot[3], ha='center', fontsize=9)
    fig.text(0.8, 0.04, properties_list_plot[4], ha='center', fontsize=9)
    """

    fig.text(0.5, 0.01, 'PROPERTIES', ha='center', fontsize=11)
    fig.text(0.17, 0.04, properties_list_plot[0], ha='center', fontsize=9)
    fig.text(0.31, 0.04, properties_list_plot[1], ha='center', fontsize=9)
    fig.text(0.45, 0.04, properties_list_plot[2], ha='center', fontsize=9)
    fig.text(0.57, 0.04, properties_list_plot[3], ha='center', fontsize=9)
    fig.text(0.7, 0.04, properties_list_plot[4], ha='center', fontsize=9)
    fig.text(0.84, 0.04, properties_list_plot[5], ha='center', fontsize=9)

    # eixo y real
    if theorical_real == 'r':
        fig.text(0.04, 0.5, 'MODELS', va='center', rotation='vertical', fontsize=11)
        fig.text(0.08, 0.85, models_list_plot[0], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.71, models_list_plot[1], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.60, models_list_plot[2], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.48, models_list_plot[3], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.37, models_list_plot[4], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.26, models_list_plot[5], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.15, models_list_plot[6], va='center', rotation='vertical', fontsize=9)

    # eixo y models
    if theorical_real == 't':
        fig.text(0.04, 0.5, 'MODELS', va='center', rotation='vertical', fontsize=11)
        fig.text(0.08, 0.8, models_list_plot[0], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.68, models_list_plot[1], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.55, models_list_plot[2], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.44, models_list_plot[3], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.32, models_list_plot[4], va='center', rotation='vertical', fontsize=9)
        fig.text(0.08, 0.2, models_list_plot[5], va='center', rotation='vertical', fontsize=9)

    #fig.legend(loc='center right', borderaxespad=0.1)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    long_type = 'models' if theorical_real == 't' else 'real'
    plt.savefig('wnew_knowledge_all_perc_' + long_type + '_' + metric + '.pdf', format='pdf', dpi=100)  # , bbox_inches="tight")






def group_knowledge_correlation_vertical(theorical_real = 't', metric='pearson'):

    if theorical_real == 't':
        models_list = models_list_t
        models_list_plot = models_list_plot_t
    elif theorical_real == 'r':
        models_list = models_list_r
        models_list_plot = models_list_plot_r

    """
        network_size = {
            'facebook': 320,
            'power': 4941,
            'econ-poli': 3915,
            'web-EPA': 4271,
            'Collins': 1622,
            'AI_interactions': 4829,
            'socfb-JohnsHopkins55': 5180,
            'er': 5000,
            'ba': 5000,
            'wax': 5000,
            'lfr': 5000,
            'lfr_m05': 5000,
            'lfr_m8': 5000
        }
        """

    network_size = {
        'facebook': 320,
        'power': 4941,
        'econ-poli': 2343,
        'web-EPA': 4253,
        'Collins': 1004,
        'bio-DM-CX': 4032,
        'AI_interactions': 4519,
        'socfb-JohnsHopkins55': 5157,
        'er': 5001,
        'ba': 5000,
        'wax': 4900,
        'lfr': 5000,
        'lfr_m05': 5000,
        'lfr_m8': 5000
    }

    fig, axs = plt.subplots(len(properties_list), len(models_list), sharex=False, sharey=True, figsize=(11, 8))
    fig.subplots_adjust(wspace=0, hspace=0)

    totplots = 0

    ax0 = 0

    prop1 = 'knowledge'

    with open('sg_wnew_results_' + prop1 + '.pkl', 'rb') as f:
        results_knowledge = pickle.load(f)

    for prop in properties_list:

        ax1 = 0

        with open('sg_wnew_results_' + prop + '.pkl', 'rb') as f:
            results_prop = pickle.load(f)

        for model in models_list:

            print(prop, model)

            result_plot = {}

            for net_number in range(TOTAL_NETS):

                for walk_size in WALK_SIZE_LIST:

                    result_plot[walk_size] = {}

                    for dynamics in dynamics_list:

                        pointsx = []
                        pointsy = []

                        list_avg_pointsx = []

                        for walk_number in range(TOTAL_WALKS):

                            for vertex in list(results_prop[model][net_number][walk_size][walk_number][dynamics].keys()):
                                pointx = results_knowledge[model][net_number][walk_size][walk_number][dynamics][vertex]
                                pointy = results_prop[model][net_number][walk_size][walk_number][dynamics][vertex]

                                pointsx.append(pointx)
                                pointsy.append(pointy)

                            list_avg_pointsx.append(pointsx[-1][0])

                        avg_point = sum(list_avg_pointsx) / len(list_avg_pointsx)
                        avg_point = avg_point / network_size[model]

                        x_corr = [x_[0] for x_ in pointsy]
                        y_corr = [y_[1] for y_ in pointsy]

                        x_corr = [x_ if math.isnan(float(x_)) is False else 0 for x_ in x_corr]
                        y_corr = [y_ if math.isnan(float(y_)) is False else 0 for y_ in y_corr]


                        p_corr, _ = pearsonr(x_corr, y_corr)
                        s_corr, _ = spearmanr(x_corr, y_corr)

                        result_plot[walk_size][dynamics] = {}

                        result_plot[walk_size][dynamics]['pearson'] = p_corr
                        result_plot[walk_size][dynamics]['spearman'] = s_corr

                        result_plot[walk_size][dynamics]['x'] = avg_point

            y_rw = []
            y_rwd = []
            y_rwid = []
            y_tsaw = []
            y_tsaw_node = []

            x_rw = []
            x_rwd = []
            x_rwid = []
            x_tsaw = []
            x_tsaw_node = []

            for w in WALK_SIZE_LIST:
                y_rw.append(result_plot[w]['rw'][metric])
                y_rwd.append(result_plot[w]['rwd'][metric])
                y_rwid.append(result_plot[w]['rwid'][metric])
                y_tsaw.append(result_plot[w]['tsaw'][metric])
                y_tsaw_node.append(result_plot[w]['tsaw_node'][metric])

                x_rw.append(result_plot[w]['rw']['x'])
                x_rwd.append(result_plot[w]['rwd']['x'])
                x_rwid.append(result_plot[w]['rwid']['x'])
                x_tsaw.append(result_plot[w]['tsaw']['x'])
                x_tsaw_node.append(result_plot[w]['tsaw_node']['x'])


                axs[ax0, ax1].plot(x_rw, y_rw, ms=3, marker='o', label='RW', color='red')
                axs[ax0, ax1].plot(x_rwd, y_rwd, ms=3, marker='s', label='RWD', color='green')
                axs[ax0, ax1].plot(x_rwid, y_rwid, ms=3, marker='v', label='RWID', color='blue')
                axs[ax0, ax1].plot(x_tsaw, y_tsaw, ms=3, marker='D', label='TSAW_Edge', color='orange')
                axs[ax0, ax1].plot(x_tsaw_node, y_tsaw_node, ms=3, marker='+', label='TSAW_Node', color='lightblue')

            axs[ax0, ax1].axhline(y=0.0, color='lightgray', linestyle='--')
            axs[ax0, ax1].axhline(y=0.5, color='lightgray', linestyle='--')
            axs[ax0, ax1].axhline(y=1.0, color='lightgray', linestyle='--')

            #axs[ax0, ax1].set_xscale('log')
            #axs[ax0, ax1].set_xticks(WALK_SIZE_LIST)
            #


            #max_xlim = max(x_rw[-1], x_rwd[-1], x_rwid[-1], x_tsaw[-1], x_tsaw_node[-1])
            #axs[ax0, ax1].set_xlim([-50, 500 + max_xlim])

            axs[ax0, ax1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axs[ax0, ax1].set_xlim([-0.05, 1.05])
            axs[ax0, ax1].set_ylim([-0.5, 1.05])




            #axs[ax0, ax1].set_xticklabels(rotation=45, fontsize=6, ha='right')

            totplots += 1

            ax1 += 1

        ax0 += 1

    print('finish')



    # eixo x
    if theorical_real == 'r':
        fig.text(0.5, 0.01, 'MODELS', ha='center', fontsize=11)
        fig.text(0.15, 0.04, models_list_plot[0], ha='center', fontsize=9)
        fig.text(0.26, 0.04, models_list_plot[1], ha='center', fontsize=9)
        fig.text(0.37, 0.04, models_list_plot[2], ha='center', fontsize=9)
        fig.text(0.48, 0.04, models_list_plot[3], ha='center', fontsize=9)
        fig.text(0.60, 0.04, models_list_plot[4], ha='center', fontsize=9)
        fig.text(0.72, 0.04, models_list_plot[5], ha='center', fontsize=9)
        fig.text(0.85, 0.04, models_list_plot[6], ha='center', fontsize=9)
    else:
        fig.text(0.5, 0.01, 'MODELS', ha='center', fontsize=11)
        fig.text(0.2, 0.04, models_list_plot[0], ha='center', fontsize=9)
        fig.text(0.32, 0.04, models_list_plot[1], ha='center', fontsize=9)
        fig.text(0.44, 0.04, models_list_plot[2], ha='center', fontsize=9)
        fig.text(0.58, 0.04, models_list_plot[3], ha='center', fontsize=9)
        fig.text(0.7, 0.04, models_list_plot[4], ha='center', fontsize=9)
        fig.text(0.83, 0.04, models_list_plot[5], ha='center', fontsize=9)

    # eixo y real
    fig.text(0.04, 0.5, 'PROPERTIES', va='center', rotation='vertical', fontsize=11)
    fig.text(0.08, 0.83, properties_list_plot[0], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.7, properties_list_plot[1], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.57, properties_list_plot[2], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.45, properties_list_plot[3], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.31, properties_list_plot[4], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.17, properties_list_plot[5], va='center', rotation='vertical', fontsize=9)

    # eixo y models
    #fig.text(0.04, 0.5, 'MODELS', va='center', rotation='vertical', fontsize=11)
    #fig.text(0.08, 0.8, models_list_plot[0], va='center', rotation='vertical', fontsize=9)
    #fig.text(0.08, 0.68, models_list_plot[1], va='center', rotation='vertical', fontsize=9)
    #fig.text(0.08, 0.55, models_list_plot[2], va='center', rotation='vertical', fontsize=9)
    #fig.text(0.08, 0.44, models_list_plot[3], va='center', rotation='vertical', fontsize=9)
    #fig.text(0.08, 0.32, models_list_plot[4], va='center', rotation='vertical', fontsize=9)
    #fig.text(0.08, 0.2, models_list_plot[5], va='center', rotation='vertical', fontsize=9)

    #fig.legend(loc='center right', borderaxespad=0.1)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    long_type = 'models' if theorical_real == 't' else 'real'

    plt.savefig('sg_wnew_knowledge_' + long_type + '_all2_' + metric + '.pdf', format='pdf', dpi=100)  # , bbox_inches="tight")



def plot_community_nmi2(measure='nmi'):

    prop = 'community_leiden'

    #measures = ['nmi', 'nmi2', 'adj_mi', 'vi', 'fmeasure', 'elscore', 'jaccard']

    with open('results_' + prop + '.pkl', 'rb') as f:
        results = pickle.load(f)

    models_list = models_list_t + models_list_r

    for model in models_list:

        result_plot = {}

        for net_number in range(TOTAL_NETS):

            for walk_size in WALK_SIZE_LIST:

                print(walk_size)

                result_plot[walk_size] = {}

                for dynamics in dynamics_list:

                    measures = []

                    for walk_number in range(TOTAL_WALKS):

                        points = []

                        for vertex in list(results[model][net_number][walk_size][walk_number][dynamics].keys()):
                            point = results[model][net_number][walk_size][walk_number][dynamics][vertex]

                            points.append(point)

                        points_reconstructed = [p[0] for p in points]
                        points_original = [p[1] for p in points]

                        if measure == 'nmi':
                            measure_result = metrics.normalized_mutual_info_score(points_reconstructed, points_original)
                        else:
                            c1 = Clustering()
                            c1.from_membership_list(points_reconstructed)
                            c2 = Clustering()
                            c2.from_membership_list(points_original)
                        if measure == 'jaccard':
                            measure_result = sim.jaccard_index(c1, c2)
                        if measure == 'nmi-clusim':
                            measure_result = sim.nmi(c1, c2)
                        if measure == 'adj_mi':
                            measure_result = sim.adj_mi(c1, c2)
                        if measure == 'vi':
                            measure_result = sim.vi(c1, c2)
                        if measure == 'fmeasure':
                            measure_result = sim.fmeasure(c1, c2)
                        if measure == 'elscore':
                            measure_result = sim.element_sim(c1, c2)

                        measures.append(measure_result)

                    avg_measures = sum(measures) / len(measures)
                    result_plot[walk_size][dynamics] = avg_measures

        y_rw = []
        y_rwd = []
        y_rwid = []
        y_tsaw = []
        y_tsaw_node = []

        fig1, ax1 = plt.subplots()

        for w in WALK_SIZE_LIST:
            y_rw.append(result_plot[w]['rw'])
            y_rwd.append(result_plot[w]['rwd'])
            y_rwid.append(result_plot[w]['rwid'])
            y_tsaw.append(result_plot[w]['tsaw'])
            y_tsaw_node.append(result_plot[w]['tsaw_node'])


        ax1.plot(WALK_SIZE_LIST, y_rw, marker='o', label='RW', color='red')
        ax1.plot(WALK_SIZE_LIST, y_rwd, marker='s', label='RWD', color='green')
        ax1.plot(WALK_SIZE_LIST, y_rwid, marker='v', label='RWID', color='blue')
        ax1.plot(WALK_SIZE_LIST, y_tsaw, marker='D', label='TSAW', color='orange')
        ax1.plot(WALK_SIZE_LIST, y_tsaw_node, marker='+', label='TSAW_NODE', color='lightblue')

        ax1.set_xscale('log')
        ax1.set_xticks(WALK_SIZE_LIST)
        # ax1.set_xticklabels(WALK_SIZE_LIST, rotation=45, ha='right')
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        plt.title(str(model).upper() + ' - ' + str(prop).upper() + ' - ' + measure)# + ' - ' + str(metric).upper())
        plt.legend()
        plt.tight_layout()

        figname = '_'.join(('testr', str(model), str('communities - leiden'), measure))

        plt.savefig(figname + '.pdf', format='pdf', dpi=500)

        plt.clf()




def plot_community_nmi3(measure='nmi'):

    prop = 'community_leiden'


    #with open('g_wnew_results_' + prop + '.pkl', 'rb') as f:
    with open('sg_wnew_results_' + prop + '.pkl', 'rb') as f:
        results = pickle.load(f)

    models_list = ['lfr_m05'] #models_list_r # + models_list_r

    for model in models_list:

        result_plot = {}

        for net_number in range(TOTAL_NETS):

            for walk_size in WALK_SIZE_LIST:

                print(walk_size)

                result_plot[walk_size] = {}

                for dynamics in dynamics_list:

                    measures = []

                    for walk_number in range(TOTAL_WALKS):

                        points = []

                        for vertex in list(results[model][net_number][walk_size][walk_number][dynamics].keys()):
                            point = results[model][net_number][walk_size][walk_number][dynamics][vertex]
                            points.append(point)

                        points_reconstructed = [p[0] for p in points]
                        points_original = [p[1] for p in points]

                        if measure == 'nmi':
                            measure_result = metrics.normalized_mutual_info_score(points_reconstructed, points_original)
                            print(measure_result)
                        else:
                            c1 = Clustering()
                            c1.from_membership_list(points_reconstructed)
                            c2 = Clustering()
                            c2.from_membership_list(points_original)
                        if measure == 'jaccard':
                            measure_result = sim.jaccard_index(c1, c2)
                        if measure == 'nmi-clusim':
                            measure_result = sim.nmi(c1, c2)
                        if measure == 'adj_mi':
                            measure_result = sim.adj_mi(c1, c2)
                        if measure == 'vi':
                            measure_result = sim.vi(c1, c2)
                        if measure == 'fmeasure':
                            measure_result = sim.fmeasure(c1, c2)
                        if measure == 'elscore':
                            measure_result = sim.element_sim(c1, c2)
                        if measure == 'ari':
                            measure_result = sim.adjrand_index(c1, c2)

                        measures.append(measure_result)

                    avg_measures = sum(measures) / len(measures)
                    result_plot[walk_size][dynamics] = avg_measures

        y_rw = []
        y_rwd = []
        y_rwid = []
        y_tsaw = []
        y_tsaw_node = []

        fig1, ax1 = plt.subplots()

        for w in WALK_SIZE_LIST:
            y_rw.append(result_plot[w]['rw'])
            y_rwd.append(result_plot[w]['rwd'])
            y_rwid.append(result_plot[w]['rwid'])
            y_tsaw.append(result_plot[w]['tsaw'])
            y_tsaw_node.append(result_plot[w]['tsaw_node'])


        ax1.plot(WALK_SIZE_LIST, y_rw, marker='o', label='RW', color='red')
        ax1.plot(WALK_SIZE_LIST, y_rwd, marker='s', label='RWD', color='green')
        ax1.plot(WALK_SIZE_LIST, y_rwid, marker='v', label='RWID', color='blue')
        ax1.plot(WALK_SIZE_LIST, y_tsaw, marker='D', label='TSAW', color='orange')
        ax1.plot(WALK_SIZE_LIST, y_tsaw_node, marker='+', label='TSAW_NODE', color='lightblue')

        ax1.set_xscale('log')
        ax1.set_xticks(WALK_SIZE_LIST)
        # ax1.set_xticklabels(WALK_SIZE_LIST, rotation=45, ha='right')
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        plt.title(str(model).upper() + ' - ' + str(prop).upper() + ' - ' + measure)# + ' - ' + str(metric).upper())
        plt.legend()
        plt.tight_layout()

        figname = '_'.join(('communities_review_g_wnew', str(model), str('communities - leiden'), measure))

        plt.savefig(figname + '.pdf', format='pdf', dpi=500)

        plt.clf()


def group_correlation_with_community(theorical_real = 't', metric='pearson'):

    if theorical_real == 't':
        models_list = models_list_t
        models_list_plot = models_list_plot_t
    elif theorical_real == 'r':
        models_list = models_list_r
        models_list_plot = models_list_plot_r

    properties_list.append('community_leiden_nmi')
    properties_list_plot.append('NMI')
    properties_list.append('community_leiden_ari')
    properties_list_plot.append('ARI')

    fig, axs = plt.subplots(len(properties_list), len(models_list), sharex=False, sharey=True, figsize=(13, 10))
    fig.subplots_adjust(wspace=0, hspace=0)

    print(len(properties_list))

    totplots = 0

    ax0 = 0

    for prop in properties_list:

        ax1 = 0

        propget = prop
        if prop == 'community_leiden_nmi' or prop == 'community_leiden_ari':
            propget = 'community_leiden'

        with open('sg_wnew_results_' + propget + '.pkl', 'rb') as f:
            results = pickle.load(f)

        for model in models_list:

            print(prop, model)

            result_plot = {}

            for net_number in range(TOTAL_NETS):

                for walk_size in WALK_SIZE_LIST:

                    result_plot[walk_size] = {}

                    for dynamics in dynamics_list:

                        points = []
                        measures = []

                        if prop == 'community_leiden_nmi' or prop == 'community_leiden_ari':
                            pcom = []

                        for walk_number in range(TOTAL_WALKS):

                            if prop == 'community_leiden_nmi' or prop == 'community_leiden_ari':
                                pcom = []


                            for vertex in list(results[model][net_number][walk_size][walk_number][dynamics].keys()):
                                point = results[model][net_number][walk_size][walk_number][dynamics][vertex]
                                if prop == 'community_leiden_nmi' or prop == 'community_leiden_ari':
                                    pcom.append(point)
                                else:
                                    points.append(point)

                            if prop == 'community_leiden_nmi':
                                points_reconstructed = [p[0] for p in pcom]
                                points_original = [p[1] for p in pcom]
                                measure_result = metrics.normalized_mutual_info_score(points_reconstructed, points_original)
                                measures.append(measure_result)
                            if prop == 'community_leiden_ari':
                                points_reconstructed = [p[0] for p in pcom]
                                points_original = [p[1] for p in pcom]
                                measure_result = metrics.adjusted_rand_score(points_reconstructed, points_original)
                                measures.append(measure_result)

                        x = [x_[0] if math.isnan(float(x_[0])) is False else 0 for x_ in points]
                        y = [y_[1] if math.isnan(float(y_[1])) is False else 0 for y_ in points]

                        result_plot[walk_size][dynamics] = {}

                        if prop != 'community_leiden_nmi' and prop != 'community_leiden_ari':

                            if metric == 'pearson':
                                p_corr, _ = pearsonr(x, y)
                                result_plot[walk_size][dynamics]['pearson'] = p_corr
                            if metric == 'spearman':
                                s_corr, _ = spearmanr(x, y)
                                result_plot[walk_size][dynamics]['spearman'] = s_corr

                        else:

                            if theorical_real == 't' and model not in ['lfr', 'lfr_m05', 'lfr_m8']:
                                avg_measures = 0
                                #result_plot[walk_size][dynamics]['community_leiden'] = avg_measures
                                result_plot[walk_size][dynamics][prop] = avg_measures
                            else:
                                avg_measures = sum(measures) / len(measures)
                                result_plot[walk_size][dynamics][prop] = avg_measures
                                #result_plot[walk_size][dynamics]['community_leiden'] = avg_measures

            y_rw = []
            y_rwd = []
            y_rwid = []
            y_tsaw = []
            y_tsaw_node = []

            for w in WALK_SIZE_LIST:
                met_plot = ''
                if prop == 'community_leiden_nmi' or prop == 'community_leiden_ari':
                    met_plot = prop
                else:
                    met_plot = metric
                y_rw.append(result_plot[w]['rw'][met_plot])
                y_rwd.append(result_plot[w]['rwd'][met_plot])
                y_rwid.append(result_plot[w]['rwid'][met_plot])
                y_tsaw.append(result_plot[w]['tsaw'][met_plot])
                y_tsaw_node.append(result_plot[w]['tsaw_node'][met_plot])

            axs[ax0, ax1].axhline(y=0.0, color='lightgray', linestyle='--')
            axs[ax0, ax1].axhline(y=0.5, color='lightgray', linestyle='--')
            axs[ax0, ax1].axhline(y=1.0, color='lightgray', linestyle='--')

            print_plot = True

            if prop == 'community_leiden_nmi' or prop == 'community_leiden_ari':
                if theorical_real == 't':
                    if model not in ['lfr', 'lfr_m05', 'lfr_m8']:
                        print_plot = False
            if print_plot:

                axs[ax0, ax1].plot(WALK_SIZE_LIST, y_rw, ms=3, marker='o', label='RW', color='red')
                axs[ax0, ax1].plot(WALK_SIZE_LIST, y_rwd, ms=3, marker='s', label='RWD', color='green')
                axs[ax0, ax1].plot(WALK_SIZE_LIST, y_rwid, ms=3, marker='v', label='RWID', color='blue')
                axs[ax0, ax1].plot(WALK_SIZE_LIST, y_tsaw, ms=3, marker='D', label='TSAW_Edge', color='orange')
                axs[ax0, ax1].plot(WALK_SIZE_LIST, y_tsaw_node, ms=3, marker='+', label='TSAW_Node', color='lightblue')

                axs[ax0, ax1].set_xscale('log')
                #axs[ax0, ax1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                #axs[ax0, ax1].set_xticklabels(WALK_SIZE_LIST, rotation=45, fontsize=6, ha='right')

            #if ax1 == 6:
            #    axs[ax0, ax1].set_xticklabels(WALK_SIZE_LIST, rotation=45, fontsize=6, ha='right')
            #axs[ax0, ax1].set_xlim([1, 60000])
            axs[ax0, ax1].set_xticks(WALK_SIZE_LIST)
            #plt.locator_params(axis='x', nbins=6)
            #axs[ax0, ax1].set_xticks([100, 400, 1000, 5000, 20000, 50000])
            axs[ax0, ax1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axs[ax0, ax1].set_xticklabels(WALK_SIZE_LIST, rotation=90, fontsize=8, ha='right')


            ax1 += 1

        ax0 += 1




    # eixo x
    fig.text(0.5, 0.01, 'Walk length', ha='center', fontsize=11)
    fig.text(0.08, 0.85, properties_list_plot[0], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.75, properties_list_plot[1], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.65, properties_list_plot[2], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.55, properties_list_plot[3], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.45, properties_list_plot[4], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.35, properties_list_plot[5], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.25, properties_list_plot[6], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.15, properties_list_plot[7], va='center', rotation='vertical', fontsize=9)

    #fig.text(0.5, 0.01, 'Walk length', ha='center', fontsize=11)
    #fig.text(0.08, 0.83, properties_list_plot[0], va='center', rotation='vertical', fontsize=9)
    #fig.text(0.08, 0.72, properties_list_plot[1], va='center', rotation='vertical', fontsize=9)
    #fig.text(0.08, 0.61, properties_list_plot[2], va='center', rotation='vertical', fontsize=9)
    #fig.text(0.08, 0.5, properties_list_plot[3], va='center', rotation='vertical', fontsize=9)
    #fig.text(0.08, 0.38, properties_list_plot[4], va='center', rotation='vertical', fontsize=9)
    #fig.text(0.08, 0.27, properties_list_plot[5], va='center', rotation='vertical', fontsize=9)
    #fig.text(0.08, 0.17, properties_list_plot[6], va='center', rotation='vertical', fontsize=9)

    # eixo y real
    if theorical_real == 'r':
        fig.text(0.04, 0.5, 'Correlation', va='center', rotation='vertical', fontsize=11)
        fig.text(0.15, 0.04, models_list_plot[0], ha='center', fontsize=9)
        fig.text(0.26, 0.04, models_list_plot[1], ha='center', fontsize=9)
        fig.text(0.37, 0.04, models_list_plot[2], ha='center', fontsize=9)
        fig.text(0.48, 0.04, models_list_plot[3], ha='center', fontsize=9)
        fig.text(0.60, 0.04, models_list_plot[4], ha='center', fontsize=9)
        fig.text(0.72, 0.04, models_list_plot[5], ha='center', fontsize=9)
        fig.text(0.85, 0.04, models_list_plot[6], ha='center', fontsize=9)
    else: # eixo  y models
        fig.text(0.04, 0.5, 'Correlation', va='center', rotation='vertical', fontsize=11)
        fig.text(0.2, 0.04, models_list_plot[0], ha='center', fontsize=9)
        fig.text(0.32, 0.04, models_list_plot[1], ha='center', fontsize=9)
        fig.text(0.44, 0.04, models_list_plot[2], ha='center', fontsize=9)
        fig.text(0.58, 0.04, models_list_plot[3], ha='center', fontsize=9)
        fig.text(0.7, 0.04, models_list_plot[4], ha='center', fontsize=9)
        fig.text(0.83, 0.04, models_list_plot[5], ha='center', fontsize=9)

    #fig.legend(loc='center right', borderaxespad=0.1)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    long_type = 'models' if theorical_real == 't' else 'real'

    plt.savefig('review_sg6_wnew_com_' + long_type + '_all_' + metric + '.pdf', format='pdf', dpi=100)  # , bbox_inches="tight")



def group_knowledge_correlation_vertical_with_community(theorical_real = 't', metric='pearson'):

    if theorical_real == 't':
        models_list = models_list_t
        models_list_plot = models_list_plot_t
    elif theorical_real == 'r':
        models_list = models_list_r
        models_list_plot = models_list_plot_r


    network_size = {
        'facebook': 320,
        'power': 4941,
        'econ-poli': 2343,
        'web-EPA': 4253,
        'Collins': 1004,
        'bio-DM-CX': 4032,
        'AI_interactions': 4519,
        'socfb-JohnsHopkins55': 5157,
        'er': 5001,
        'ba': 5000,
        'wax': 4900,
        'lfr': 5000,
        'lfr_m05': 5000,
        'lfr_m8': 5000
    }

    #properties_list.append('community_leiden')
    #properties_list_plot.append('NMI')

    fig, axs = plt.subplots(len(properties_list), len(models_list), sharex=False, sharey=True, figsize=(11, 8))
    fig.subplots_adjust(wspace=0, hspace=0)

    totplots = 0

    ax0 = 0

    prop1 = 'knowledge'

    with open('sg_wnew_results_' + prop1 + '.pkl', 'rb') as f:
        results_knowledge = pickle.load(f)

    for prop in properties_list:

        ax1 = 0

        with open('sg_wnew_results_' + prop + '.pkl', 'rb') as f:
            results_prop = pickle.load(f)

        for model in models_list:

            print(prop, model)

            result_plot = {}

            for net_number in range(TOTAL_NETS):

                for walk_size in WALK_SIZE_LIST:

                    result_plot[walk_size] = {}

                    for dynamics in dynamics_list:

                        pointsx = []
                        pointsy = []

                        list_avg_pointsx = []

                        measures = []

                        if prop == 'community_leiden':
                            pcomy = []

                        for walk_number in range(TOTAL_WALKS):

                            if prop == 'community_leiden':
                                pcomy = []

                            for vertex in list(results_prop[model][net_number][walk_size][walk_number][dynamics].keys()):
                                pointx = results_knowledge[model][net_number][walk_size][walk_number][dynamics][vertex]
                                pointy = results_prop[model][net_number][walk_size][walk_number][dynamics][vertex]

                                pointsx.append(pointx)
                                pointsy.append(pointy)

                                if prop == 'community_leiden':
                                    pcomy.append(pointy)


                            #atencao:
                            #pointx = x comum: tamanho caminhada
                            #pointy = par original, reconstruida

                            if prop == 'community_leiden':
                                points_reconstructed = [p[0] for p in pcom]
                                points_original = [p[1] for p in pcom]
                                measure_result = metrics.normalized_mutual_info_score(points_reconstructed, points_original)
                                measures.append(measure_result)
                                print(measure_result)

                            list_avg_pointsx.append(pointsx[-1][0])

                        avg_point = sum(list_avg_pointsx) / len(list_avg_pointsx)
                        avg_point = avg_point / network_size[model]

                        x_corr = [x_[0] for x_ in pointsy]
                        y_corr = [y_[1] for y_ in pointsy]

                        x_corr = [x_ if math.isnan(float(x_)) is False else 0 for x_ in x_corr]
                        y_corr = [y_ if math.isnan(float(y_)) is False else 0 for y_ in y_corr]


                        p_corr, _ = pearsonr(x_corr, y_corr)
                        s_corr, _ = spearmanr(x_corr, y_corr)

                        result_plot[walk_size][dynamics] = {}

                        result_plot[walk_size][dynamics]['pearson'] = p_corr
                        result_plot[walk_size][dynamics]['spearman'] = s_corr

                        result_plot[walk_size][dynamics]['x'] = avg_point

            y_rw = []
            y_rwd = []
            y_rwid = []
            y_tsaw = []
            y_tsaw_node = []

            x_rw = []
            x_rwd = []
            x_rwid = []
            x_tsaw = []
            x_tsaw_node = []

            for w in WALK_SIZE_LIST:
                y_rw.append(result_plot[w]['rw'][metric])
                y_rwd.append(result_plot[w]['rwd'][metric])
                y_rwid.append(result_plot[w]['rwid'][metric])
                y_tsaw.append(result_plot[w]['tsaw'][metric])
                y_tsaw_node.append(result_plot[w]['tsaw_node'][metric])

                x_rw.append(result_plot[w]['rw']['x'])
                x_rwd.append(result_plot[w]['rwd']['x'])
                x_rwid.append(result_plot[w]['rwid']['x'])
                x_tsaw.append(result_plot[w]['tsaw']['x'])
                x_tsaw_node.append(result_plot[w]['tsaw_node']['x'])

            axs[ax0, ax1].plot(x_rw, y_rw, ms=3, marker='o', label='RW', color='red')
            axs[ax0, ax1].plot(x_rwd, y_rwd, ms=3, marker='s', label='RWD', color='green')
            axs[ax0, ax1].plot(x_rwid, y_rwid, ms=3, marker='v', label='RWID', color='blue')
            axs[ax0, ax1].plot(x_tsaw, y_tsaw, ms=3, marker='D', label='TSAW_Edge', color='orange')
            axs[ax0, ax1].plot(x_tsaw_node, y_tsaw_node, ms=3, marker='+', label='TSAW_Node', color='lightblue')

            axs[ax0, ax1].axhline(y=0.0, color='lightgray', linestyle='--')
            axs[ax0, ax1].axhline(y=0.5, color='lightgray', linestyle='--')
            axs[ax0, ax1].axhline(y=1.0, color='lightgray', linestyle='--')

            #axs[ax0, ax1].set_xscale('log')
            #axs[ax0, ax1].set_xticks(WALK_SIZE_LIST)
            #


            #max_xlim = max(x_rw[-1], x_rwd[-1], x_rwid[-1], x_tsaw[-1], x_tsaw_node[-1])
            #axs[ax0, ax1].set_xlim([-50, 500 + max_xlim])

            axs[ax0, ax1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axs[ax0, ax1].set_xlim([-0.05, 1.05])
            axs[ax0, ax1].set_ylim([-0.5, 1.05])




            #axs[ax0, ax1].set_xticklabels(rotation=45, fontsize=6, ha='right')

            totplots += 1

            ax1 += 1

        ax0 += 1

    print('finish')



    # eixo x
    if theorical_real == 'r':
        fig.text(0.5, 0.01, 'Discovered nodes (%)', ha='center', fontsize=11)
        fig.text(0.15, 0.04, models_list_plot[0], ha='center', fontsize=9)
        fig.text(0.26, 0.04, models_list_plot[1], ha='center', fontsize=9)
        fig.text(0.37, 0.04, models_list_plot[2], ha='center', fontsize=9)
        fig.text(0.48, 0.04, models_list_plot[3], ha='center', fontsize=9)
        fig.text(0.60, 0.04, models_list_plot[4], ha='center', fontsize=9)
        fig.text(0.72, 0.04, models_list_plot[5], ha='center', fontsize=9)
        fig.text(0.85, 0.04, models_list_plot[6], ha='center', fontsize=9)
    else:
        fig.text(0.5, 0.01, 'Discovered nodes (%)', ha='center', fontsize=11)
        fig.text(0.2, 0.04, models_list_plot[0], ha='center', fontsize=9)
        fig.text(0.32, 0.04, models_list_plot[1], ha='center', fontsize=9)
        fig.text(0.44, 0.04, models_list_plot[2], ha='center', fontsize=9)
        fig.text(0.58, 0.04, models_list_plot[3], ha='center', fontsize=9)
        fig.text(0.7, 0.04, models_list_plot[4], ha='center', fontsize=9)
        fig.text(0.83, 0.04, models_list_plot[5], ha='center', fontsize=9)

    # eixo y real
    fig.text(0.04, 0.5, 'Correlation', va='center', rotation='vertical', fontsize=11)
    fig.text(0.08, 0.83, properties_list_plot[0], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.7, properties_list_plot[1], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.57, properties_list_plot[2], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.45, properties_list_plot[3], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.31, properties_list_plot[4], va='center', rotation='vertical', fontsize=9)
    fig.text(0.08, 0.17, properties_list_plot[5], va='center', rotation='vertical', fontsize=9)

    # eixo y models
    #fig.text(0.04, 0.5, 'MODELS', va='center', rotation='vertical', fontsize=11)
    #fig.text(0.08, 0.8, models_list_plot[0], va='center', rotation='vertical', fontsize=9)
    #fig.text(0.08, 0.68, models_list_plot[1], va='center', rotation='vertical', fontsize=9)
    #fig.text(0.08, 0.55, models_list_plot[2], va='center', rotation='vertical', fontsize=9)
    #fig.text(0.08, 0.44, models_list_plot[3], va='center', rotation='vertical', fontsize=9)
    #fig.text(0.08, 0.32, models_list_plot[4], va='center', rotation='vertical', fontsize=9)
    #fig.text(0.08, 0.2, models_list_plot[5], va='center', rotation='vertical', fontsize=9)

    #fig.legend(loc='center right', borderaxespad=0.1)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')

    long_type = 'models' if theorical_real == 't' else 'real'

    plt.savefig('sg5_wnew_knowledge_' + long_type + '_all2_' + metric + '.pdf', format='pdf', dpi=100)  # , bbox_inches="tight")





#plot_community_nmi3(measure='ari')

#plot_graph1_v5()

group_correlation_with_community('t', 'pearson')
#group_correlation_with_community('r', 'pearson')

#group_knowledge_correlation_vertical_with_community('t', 'pearson')
#group_knowledge_correlation_vertical_with_community('r', 'pearson')

print('done')
import pickle

models_to_join = ['er', 'ba', 'wax', 'lfr', 'facebook', 'power', 'econ-poli', 'web-EPA', 'bio-DM-CX', 'lfr_m05', 'AI_interactions', 'socfb-JohnsHopkins55', 'lfr_m8']
TOTAL_NETS = list(range(1))
TOTAL_WALKS = list(range(20))

walks_to_join = [100, 200, 400, 500, 600, 800, 1000, 2000, 5000, 20000, 50000]
#properties_to_join = ['betweenness']
properties_to_join = ['degree', 'closeness', 'eccentricity', 'cc', 'coreness', 'knowledge', 'community_leiden', 'betweenness']


for prop in properties_to_join:

    results = {}

    for model in models_to_join:

        results[model] = {}

        for net_number in TOTAL_NETS:

            results[model][net_number] = {}

            for walk_size in walks_to_join:

                filename_to_read = '_'.join(('sg_wnew_', str(model), str(net_number), str(walk_size), str(prop))) + '.pkl'
                with open('data/' + filename_to_read, 'rb') as f:
                    results[model][net_number][walk_size] = pickle.load(f)

    with open('sg_wnew_results_' + prop + '.pkl', 'wb') as f:
        pickle.dump(results, f)

print('terminou')
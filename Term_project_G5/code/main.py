import argparse
import networkx as nx
import time

import function

parser = argparse.ArgumentParser()
parser.add_argument('--network', default="../data/TC1/TC1-1/1-1.dat", help='a network file name')
args = parser.parse_args()
print('network', args.network)
G = nx.read_edgelist(args.network)
print(G)

# find seed nodes
s_time = time.time()
n = G.number_of_nodes()
seeds = function.find_seed(G, 0.1, n, 0.001)
end_time = time.time() - s_time

# community detection
communities = []
methods = ['clustering_coefficient', 'avg_shortest_path', 'density']
resolutions = [0.5, 1.0, 1.5, 2.0]
for method in methods:
    for resolution1 in resolutions:
        for resolution2 in resolutions:
            community_dict, run_time, modularity, size, see_num = function.top_add_detection_test(seeds, G, method, resolution1, resolution2)
            communities.append([community_dict, run_time, modularity, size, see_num])
sorted_communities = sorted(communities, key=lambda x: x[2], reverse=True)
final_community = sorted_communities[int(len(sorted_communities)*(1/3))]
file_path = (args.network).replace('.dat', f'.cmty')
with open(file_path, 'w') as file:
    for key, value in final_community[0].items():
        file.write(f'{key} {value}\n')


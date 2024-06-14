import time
import numpy as np
import community as community_louvain
from copy import deepcopy
import networkx as nx
from itertools import combinations

import kpcore

def find_seed(G, p_value, n, min_ratio):
    seeds = {}
    G_ = deepcopy(G)

    # k 범위 설정
    min_k = min(nx.core_number(G).values())
    max_k = max(nx.core_number(G).values())
    i = 1 # seed_id

    while True:
        core, k, p = find_smallest_kp_core(G_, min_k, max_k, p_value, n, min_ratio)
        if core == None: break

        # 연결된 구성 요소 찾기
        connected_components = list(nx.connected_components(core))

        # subgraph 생성
        subgraphs = [core.subgraph(component).copy() for component in connected_components]
        for subgraph in subgraphs:
            # nested seed node 처리
            if set(seeds.keys()).intersection(subgraph.nodes()):
                G_ = connect_external_nodes(G_, set(seeds.keys()).intersection(subgraph.nodes()))
                continue
            s_id = f"s_{i}"
            seeds[s_id] = {'nested':[], 'variables':[], 'nodes':[], 'edges':[]}
            seeds[s_id]['variables'] = [k,p]
            seeds[s_id]['nodes'] = subgraph.nodes()
            seeds[s_id]['edges'] = subgraph.edges()
            average_clustering = nx.average_clustering(subgraph)
            seeds[s_id]['clustering_coefficient'] = average_clustering
            average_shortest_path_length = nx.average_shortest_path_length(subgraph)
            seeds[s_id]['avg_shortest_path'] = average_shortest_path_length
            density = nx.density(subgraph)
            seeds[s_id]['density'] = density
            seeds[s_id]['seed_size'] = subgraph.number_of_nodes()

            G_ = subgraph_to_seed(G_, subgraph, s_id)
            i += 1
        if nx.number_connected_components(G_) == G_.number_of_nodes():
            break

    return seeds

def find_smallest_kp_core(G, min_k, max_k, p_value, n, min_ratio):
    G_ = deepcopy(G)
    smallest_kp = None
    smallest_size = float('inf')
    best_p = 0
    best_k = 0
    k_min = min_k
    k_max = max_k

    while k_min < k_max:
        k = (k_min+k_max) // 2
        all_zero = True

        for p in np.arange(p_value, 1.00001, p_value):
            p = round(p, 3)
            kp_core = kpcore.run(G_, k, p)  # Assuming kpcore.run() is your function to compute the (k,p)-core

            if kp_core.number_of_nodes() == 0 or kp_core.number_of_edges() == 0:
                continue  # Skip if kp_core is empty, no need to break

            all_zero = False

            current_size = kp_core.number_of_nodes()
            if current_size/n >= min_ratio and current_size <= smallest_size and kp_core.number_of_edges() != 0:
                smallest_kp = kp_core
                smallest_size = current_size
                best_k = k
                best_p = p

        if all_zero or current_size/n <= min_ratio:
            k_max = k-1 # Reduce the search space to lower k values

        else:
            k_min = k+1 # Increase the search space to higher k values


    return smallest_kp, best_k, best_p

def connect_external_nodes(graph, seed_set):
    # 시드와 연결된 외부 노드들을 찾기
    external_nodes = set()
    for node in seed_set:
        if node in graph:
            neighbors = set(graph.neighbors(node))
            external_nodes.update(neighbors)

    # external_nodes에서 서브그래프 노드 제거
    external_nodes = external_nodes - seed_set

    # 그래프에서 서브그래프 노드를 제거
    graph.remove_nodes_from(seed_set)

    # 외부 노드 쌍의 최단 경로 길이를 저장할 리스트
    edge_candidates = []
    for u, v in combinations(external_nodes, 2):
        if graph.has_edge(u, v):
            continue
        try:
            length = nx.shortest_path_length(graph, source=u, target=v)
            edge_candidates.append((length, u, v))
        except (nx.NetworkXNoPath, KeyError):
            continue

    # 최단 경로 길이 순으로 정렬
    edge_candidates.sort()

    # 각 외부 노드에 추가된 엣지 수를 기록할 딕셔너리
    node_edge_count = {node: 0 for node in external_nodes}

    # 가장 짧은 경로부터 엣지 추가
    added_edges = []
    for length, u, v in edge_candidates:
        # u 또는 v에 이미 엣지가 추가되었다면 건너뜀
        if node_edge_count[u] >= 1 or node_edge_count[v] >= 1:
            continue
        graph.add_edge(u, v)
        added_edges.append((u, v))
        node_edge_count[u] += 1
        node_edge_count[v] += 1

    return graph

def subgraph_to_seed(G, subgraph, s_id, remove=True):
    G.add_node(s_id)
    for node in subgraph.nodes():
        for neighbor in list(G.neighbors(node)):  # Use list to avoid runtime error due to dynamic changes
            if neighbor not in subgraph.nodes():
                G.add_edge(s_id, neighbor)
    if remove:
        G.remove_nodes_from(subgraph.nodes())

    return G

def top_add_detection_test(seeds_data, G, method, resolution1, resolution2):
    s_time = time.time()
    community, size, seed_num = top_seed(G, seeds_data, method)
    all_nodes = set(G.nodes())
    assigned_nodes = set(community.keys())
    unassigned_nodes = all_nodes - assigned_nodes
    first_time = time.time() - s_time
    community_dict, second_time, modularity = add_nodes_to_communities(G, community, unassigned_nodes, resolution1, resolution2, first_time)

    return community_dict, first_time+second_time, modularity, size, seed_num

def top_seed(G, seed, method):
    size = 0
    seeds_data = find_top_seed(G, seed, method)
    for id, info in seeds_data.items():
        size += info['seed_size']

    community_dict = {}
    # Assigning community IDs to nodes
    community_id = 0
    for seed_id, seed_info in seeds_data.items():
        nodes = seed_info['nodes']
        for node in nodes:
            community_dict[node] = str(community_id)
        community_id += 1
    return community_dict, size, len(seeds_data)

def find_top_seed(G, seeds, method, threshold=0.5):
    # 시드를 method 값 기준으로 오름차순 정렬
    sorted_seeds = sorted(seeds.items(), key=lambda item: item[1][method], reverse=True)
    if len(sorted_seeds) == 1:
        return dict(sorted_seeds)

    # 정렬된 시드의 method 값을 리스트로 추출
    values = [item[1][method] for item in sorted_seeds]

    eps = np.finfo(float).eps  # 작은 값 (머신 엡실론)

    # 변화율 계산
    rates_of_change = np.diff(values) / (values[:-1] + eps)

    # 변화율이 임계값을 초과하는 첫 번째 지점 찾기
    for idx, rate in enumerate(rates_of_change):
        if abs(rate) >= threshold:
            return dict(sorted_seeds[:idx+1])
        elif idx >= int(G.number_of_nodes()**0.5):
            return dict(sorted_seeds[:int(G.number_of_nodes()**0.5)+1])
    return dict(sorted_seeds[:int(G.number_of_nodes()**0.5)+1])

def add_nodes_to_communities(graph, community_dict, new_nodes, resolution1, resolution2, first_time):
    # 기존 커뮤니티 ID를 정수로 변환하여 최대값 찾기
    int_community_values = [int(val) for val in community_dict.values()]
    max_existing_community = max(int_community_values, default=0)
    new_community_id = max_existing_community + 1

    s_time = time.time()
    for new_node in new_nodes:
         # 새로운 노드의 이웃 노드들의 커뮤니티 찾기
         neighbors = set(graph.neighbors(new_node))
         neighbor_communities = [community_dict[n] for n in neighbors if n in community_dict]

         if neighbor_communities:
             # 새로운 노드를 이웃의 가장 흔한 커뮤니티에 할당
             most_common_community = max(set(neighbor_communities), key=neighbor_communities.count)
             community_dict[new_node] = most_common_community
         else:
             # 이웃 중 커뮤니티에 속한 노드가 없으면 새로운 커뮤니티에 할당
             community_dict[new_node] = str(new_community_id)
             new_community_id += 1

     # 전체 그래프에 대해 라벨 전파 수행
    initial_partition = label_propagation(graph, community_dict)

    # 모듈러리티 최적화 수행
    final_partition = community_louvain.best_partition(graph, initial_partition, resolution=resolution1)

    refined_partition = additional_detection_within_communities(graph, final_partition, resolution2, 0.3, 0.5)
    end_time = time.time() - s_time
    modularity = community_louvain.modularity(refined_partition, graph)

    return refined_partition, end_time, modularity

def label_propagation(graph, initial_partition):
    partition = initial_partition.copy()
    nodes = list(graph.nodes())
    np.random.shuffle(nodes)
    for _ in range(100):  # 최대 100번 반복
        changes = 0
        for node in nodes:
            neighbors = list(graph.neighbors(node))
            if neighbors:
                neighbor_labels = [partition[neighbor] for neighbor in neighbors if neighbor in partition]
                if neighbor_labels:
                    most_common_label = max(set(neighbor_labels), key=neighbor_labels.count)
                    if partition[node] != most_common_label:
                        partition[node] = most_common_label
                        changes += 1
        if changes == 0:
            break
    return partition

def additional_detection_within_communities(graph, community_dict, resolution, min_t, max_t):
    min_threshold = graph.number_of_nodes() ** min_t
    max_threshold = graph.number_of_nodes() ** max_t
    new_community_dict = community_dict.copy()
    max_label = max(community_dict.values()) + 1

    for community in set(community_dict.values()):
        community_nodes = [node for node in community_dict if community_dict[node] == community]
        if len(community_nodes) > max_threshold:
            subgraph = graph.subgraph(community_nodes)
            sub_partition = community_louvain.best_partition(subgraph, resolution=resolution)
            for node, sub_community in sub_partition.items():
                if sub_community == 0:
                    new_community_dict[node] = community
                else:
                    new_community_dict[node] = max_label + sub_community
            max_label += max(sub_partition.values()) + 1
        elif len(community_nodes) < min_threshold:
            # 라벨 전파를 통해 병합
            for node in community_nodes:
                neighbors = list(graph.neighbors(node))
                neighbor_communities = [new_community_dict[n] for n in neighbors if
                                        n in new_community_dict and new_community_dict[n] != community]
                if neighbor_communities:
                    most_common_community = max(set(neighbor_communities), key=neighbor_communities.count)
                    new_community_dict[node] = most_common_community
                else:
                    possible_communities = [comm for comm in set(new_community_dict.values()) if
                                            len([n for n in new_community_dict if
                                                 new_community_dict[n] == comm]) >= min_threshold]
                    if possible_communities:
                        new_community_dict[node] = np.random.choice(possible_communities)
                    else:
                        new_community_dict[node] = max(new_community_dict.values()) + 1
            # 병합된 커뮤니티에 대해 라벨 전파 실행
            new_community_dict = label_propagation(graph, new_community_dict)

    return new_community_dict
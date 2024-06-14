import time

import networkx as nx
from math import ceil

def run(G, k, p):
    # Copy the graph to avoid modifying the original graph
    G_prime = G.copy()

    i_time = time.time()

    # Initialize the combined degree threshold for each vertex
    thresholds = {v: max(k, ceil(p * G.degree(v))) for v in G_prime.nodes}

    # Initialize the queue for vertices to be deleted
    queue = [v for v in G_prime.nodes if G_prime.degree(v) < thresholds[v]]

    # Set to keep track of vertices' status
    in_queue = set(queue)

    while queue:
        v = queue.pop(0)

        if v not in G_prime:
            continue

        # Get the neighbors before removing the node
        neighbors = list(G_prime.neighbors(v))

        # Remove the vertex and its edges
        G_prime.remove_node(v)
        in_queue.discard(v)

        # Check the neighbors of the removed vertex
        for neighbor in neighbors:
            if neighbor in G_prime and G_prime.degree(neighbor) < thresholds[neighbor]:
                if neighbor not in in_queue:
                    queue.append(neighbor)
                    in_queue.add(neighbor)

    run_time = time.time()-i_time
    # connected_components = [G_prime.subgraph(c).copy() for c in nx.connected_components(G_prime)]

    return G_prime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def round_adj(adjs):
    """Round values in adjacency matrices to 0 or 1

    :param adjs: Array with shape (M,N,N) or (N,N) if unbatched
    """

    # Turn Jax array into numpy array
    rounded_adjs = np.array(adjs)

    # Set every value larger than 0.5 to 1 and else 0
    rounded_adjs[rounded_adjs > 0.5] = 1
    rounded_adjs[rounded_adjs <= 0.5] = 0

    return rounded_adjs


def adj_to_graph(adjs, batch=True):
    """Turn Array of adjacency matrices to list of networkx graphs

    :param adjs: Array of shape (batch_size,N_nodes,N_nodes) if batch=True, else (N_nodes,N_nodes)
    :param batch: Boolean, do specify if the input is batched or not
    :return graphs: List with each element being a networkx Graph
    """

    if batch:
        graphs = [nx.from_numpy_array(adj) for adj in adjs]
    else:
        graphs = [nx.from_numpy_array(adjs)]

    return graphs


def color_mapping(graph, num_nodes=13, max_degree=4):
    """For a given graph, color each node red which has has higher than max_degree and else blue

    :param graph: Networkx graph
    :param max_degree: upper limit, from which on to change the node color
    :return color_map: Color for each node
    """

    # Initialize array with color blue and length of num_nodes
    color_map = np.array(["blue"] * num_nodes)

    # Get list of nodes, with degree larger than max_degree
    degree_dict = dict(
        graph.degree
    )  # Dictionary with node as key and its degree as value

    nodes = np.array(list(degree_dict.keys()))
    degrees = np.array(list(degree_dict.values()))

    new_color_nodes = nodes[degrees > max_degree]
    color_map[new_color_nodes] = "red"

    return color_map


def plotting(graphs, num_nodes=13, max_degree=4):
    """
    Plot a list of graphs in a grid

    Parameters
    ---
    graphs: np.darray
        List of networkx graphs

    num_nodes: int
        Number of nodes in the graph

    max_degree: int
        Upper limit, from which on to change the node color

    Returns
    ---
    None

    """

    # Determine grid size
    grid_num = int(np.sqrt(len(graphs))) + 1

    fig, ax = plt.subplots(grid_num, grid_num)

    for graph, axis in zip(graphs, ax.ravel()):
        color_map = color_mapping(graph, max_degree, num_nodes)
        nx.draw(graph, node_color=color_map, ax=axis, node_size=100 / grid_num)

    for i in range(len(graphs), len(ax.ravel())):
        ax.ravel()[i].remove()

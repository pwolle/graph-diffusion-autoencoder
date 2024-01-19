import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import jax.numpy as jnp


def round_adj(
    adjs: jnp.array,
) -> jnp.array:
    """
    Round each element in adjacency matrices to 0 or 1

    Parameters
    ---
    adjs: jnp.array
        Jax array with shape (batch_size,Num_nodes,Num_nodes), representing a batch of noisy adjacency matrices.

    Returns
    ---
    jnp.array
        Rounded noisy adjacency matrix such that each element is 0 or 1
    """

    # Turn every value in adjs which is <a_min to 0 and every value > a_max to 1
    rounded_adjs = jnp.clip(adjs, a_min=0, a_max=1)

    # Round the remaining values between 0-1 to 0 or 1. 0.5 is rounded down to 0.
    rounded_adjs = jnp.round(rounded_adjs)

    return rounded_adjs


def adj_to_graph(
    adjs: jnp.array,
) -> nx.Graph:
    """
    Turn each adjacency matrix in the batch into an networkx graph and save into a list (len(list) == batch size)

    Parameters
    ---
    adjs: jnp.array
        Jax array of shape (batch_size,Num_nodes,Num_nodes)

    Returns
    ---
    list[nx.Graph]
        List with each element being a networkx Graph
    """

    return [nx.from_numpy_array(adj) for adj in adjs]


def color_mapping(
    graph: nx.Graph,
    max_degree: int,
) -> np.array:
    """
    Create a color map for each node in the graph. If the graph is connected,
    then color the node red which has a higher degree than max_degree and else blue.
    If the graph is unconnected, then color the whole graph orange.

    Parameters
    ---
    graph: nx.Graph
        Networkx graph

    max_degree: int
        Sets an upper limit for the degree of a node.
        Nodes in a connected graph with a degree larger than max_degree are colored red

    Returns
    ---
    tuple(np.array,str)
        np.array
            Numpy array with strings as elements. Each string correspond to one color of one node
        str
            Gives the status of graph, whether it is disconnected, or has nodes with too high degree, or none of them (normal).
    """

    # Find the number of nodes of given graph
    num_nodes = len(graph.nodes)

    # Sets status of graph, whether it is disconnected, or has nodes with too high degree, or none of them (normal).
    status = "normal"

    # If graph is disconnected, then color whole graph orange
    if not nx.is_connected(graph):
        status = "disconnected"
        return np.array(["orange"] * num_nodes), status

    # Initialize array with color blue and length of num_nodes
    color_map = np.array(["blue"] * num_nodes)

    # Dictionary with node as key and its degree as value
    degree_dict = dict(graph.degree)

    nodes = np.array(list(degree_dict.keys()))
    degrees = np.array(list(degree_dict.values()))

    # Find nodes whose degree is larger than max_degree
    new_color_nodes = nodes[degrees > max_degree]

    if len(new_color_nodes) > 0:
        status = "too_high"

    # Color these nodes red
    color_map[new_color_nodes] = "red"

    return color_map, status


def plotting(
    graphs: list[nx.Graph],
    max_degree: int,
    file_name: str,
) -> None:
    """
    Plot a list of graphs in a grid

    Parameters
    ---
    graphs: np.darray
        List of networkx graphs

    max_degree: int
        Upper limit, from which on to change the node color

    file_name: str
        Name of the pdf file (file_name.pdf), in which the plotted graphs are saved.

    Returns
    ---
    None

    """

    # Determine grid size
    grid_num = int(np.sqrt(len(graphs))) + 1

    # Save number of graphs, where it is disconnected, or has nodes with too high degree, or none of both (normal).
    stat_count = {"too_high": 0, "disconnected": 0, "normal": 0}

    fig, ax = plt.subplots(grid_num, grid_num)

    # Plot each graph into the grid
    for graph, axis in zip(graphs, ax.ravel()):
        color_map, status = color_mapping(graph, max_degree)

        # Add +1 to the status of the current graph
        stat_count[status] += 1

        nx.draw(graph, node_color=color_map, ax=axis, node_size=100 / grid_num)

    # Delete unused axis
    for i in range(len(graphs), len(ax.ravel())):
        ax.ravel()[i].remove()

    fig.suptitle(
        f"Normal: {stat_count['normal']}, Too high degree: {stat_count['too_high']}, Disconnected: {stat_count['disconnected']}"
    )
    plt.tight_layout()
    # Save file
    plt.savefig(f"{file_name}.pdf")


def plot(
    adjs: jnp.array,
    max_degree: int,
    file_name: str,
) -> None:
    graphs = adj_to_graph(adjs)
    plotting(graphs, max_degree, file_name)

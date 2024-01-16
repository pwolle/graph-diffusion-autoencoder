from sample_symmetric import sample, score_function
from models import BinaryEdgesModel
from plots import round_adj, adj_to_graph, color_mapping, plotting
import jax
import jax.random as jrandom
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt


key = jrandom.PRNGKey(0)
key, modul_key = jrandom.split(key)
model = BinaryEdgesModel(modul_key, nlayer=3, dim=128, dim_at=8)
model = model.load_leaves("model_2024-01-15 13:54:48.npz")


adjacency_example = jnp.zeros((10, 10))
sigma_edample = jnp.ones(())

output = model(adjacency_example, sigma_edample)
print(output[2, 3])

score = score_function(model)
# print(score(adjacency_example, sigma_edample))

key, sample_key = jrandom.split(key)

sampled = sample(
    sigmas=jnp.array([1.0, 0.5, 0.1]),
    score=score,
    num_iterations=100,
    step_size=1e-4,
    shape=(10, 10),
    key=sample_key,
)
# print(sampled)
rounded_sampled = round_adj(sampled)
print(rounded_sampled)
sampled_graph = adj_to_graph(rounded_sampled, batch=False)
print(sampled_graph)
nx.draw(sampled_graph[0])
plt.draw()
# plotting(sampled_graph, num_nodes=10, max_degree=4)

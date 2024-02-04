import flarejax as fj
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats

EPS_DEFAULT = 1e-4


class Linear(fj.Module):
    def __init__(self, key, dim_in, dim, bias=False):
        self.dim_in = dim_in
        self.dim = dim

        w = jrandom.normal(key, (dim_in, dim), dtype=jnp.float32)
        w = w / jnp.sqrt(dim_in)

        self.w = fj.Param(w)

        b = jnp.zeros((dim,), dtype=jnp.float32) if bias else None
        self.b = fj.Param(b)

    def __call__(self, x):
        x = x @ self.w.data

        if self.b.data is None:
            return x

        return x + self.b.data


def layer_norm(x, axis=-1, eps=EPS_DEFAULT):
    mean = jnp.mean(x, axis=axis, keepdims=True)
    var = jnp.var(x, axis=axis, keepdims=True)
    inv = lax.rsqrt(var + eps)
    return (x - mean) * inv


class LayerNorm(fj.Module):
    def __init__(self, dim, eps=EPS_DEFAULT):
        self.dim = dim
        self.eps = eps

        self.w = fj.Param(jnp.ones((dim,)))
        self.b = fj.Param(jnp.zeros((dim,)))

    def __call__(self, x):
        x = layer_norm(x)
        x = x * self.w.data + self.b.data
        return x


class Sequential(fj.ModuleList):
    def __call__(self, x):
        for module in self.modules:
            x = module(x)

        return x


def fourier_features(x, n=32, eps=EPS_DEFAULT):
    assert n % 2 == 0

    i = jnp.linspace(
        jnp.maximum(-n / 4, jnp.log2(eps)),
        jnp.minimum(n / 4, -jnp.log2(eps)),
        n // 2,
    )
    i = 2 * jnp.pi * 2**i

    return jnp.concatenate(
        [
            jnp.sin(x * i),
            jnp.cos(x * i),
        ],
        axis=-1,
    )


def ratio_encoding(binary_with_noise, sigma, eps=EPS_DEFAULT):
    p0 = jstats.norm.pdf(binary_with_noise, loc=0, scale=sigma)
    p1 = jstats.norm.pdf(binary_with_noise, loc=1, scale=sigma)
    return (p1 + eps) / (p0 + p1 + eps)


def set_diagonal(x, value):
    assert x.ndim == 2, f"ndim != 2, got shape {x.shape}"
    assert x.shape[0] == x.shape[1]

    n = x.shape[-1]
    r = jnp.arange(n, dtype=jnp.int32)
    x = x.at[r, r].set(value)
    return x


class InputLayer(fj.Module):
    def __init__(self, key, dim, prior=0.5):
        self.dim = dim
        self.prior = prior

        key1, key2, key3, key_virtual = jrandom.split(key, 4)
        self.mlp = Sequential(
            Linear(key1, 2 * dim, dim),
            LayerNorm(dim),
            jnn.relu,
            Linear(key2, dim, dim),
            LayerNorm(dim),
            jnn.relu,
            Linear(key3, dim, dim),
        )
        self.virtual = AddVirtualNode(key_virtual, dim)

    def __call__(self, noisy_adjacency, sigma):
        adjacency = ratio_encoding(noisy_adjacency, sigma, self.prior)
        adjacency = set_diagonal(noisy_adjacency, 1)

        degree = jnp.sum(adjacency, axis=-1, keepdims=True) - 5
        degree = fourier_features(degree, self.dim)

        sigma_encoded = fourier_features(sigma, self.dim)[None]
        sigma_encoded = jnp.repeat(sigma_encoded, degree.shape[-2], axis=-2)

        vertex_features = jnp.concatenate([degree, sigma_encoded], axis=-1)
        vertex_features = self.mlp(vertex_features)

        adjacency, vertex_features = self.virtual(adjacency, vertex_features)
        return adjacency, vertex_features


class AddVirtualNode(fj.Module):
    def __init__(self, key, dim):
        self.dim = dim
        self.param = fj.Param(jrandom.normal(key, (1, dim), dtype=jnp.float32))

    def __call__(self, adjacency, vertex_features):
        size = adjacency.shape[0]
        adjacency_new = jnp.ones(
            (size + 1, size + 1),
            dtype=jnp.float32,
        )
        adjacency_new = adjacency_new.at[:size, :size].set(adjacency)

        vertex_features_new = jnp.concatenate(
            [vertex_features, self.param.data],
            axis=0,
        )
        return adjacency_new, vertex_features_new


def remove_virtual_node(adjacency, vertex_features):
    return adjacency[:-1, :-1], vertex_features[:-1]


class GCNLayer(fj.Module):
    def __init__(self, key, dim_in, dim):
        self.dim_in = dim_in
        self.dim = dim

        self.linear1 = Linear(key, dim_in, dim)
        self.linear2 = Linear(key, dim, dim)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

    def __call__(self, adjacency, vertex_features):
        residual = vertex_features
        vertex_features = self.linear1(vertex_features)
        vertex_features = self.norm1(vertex_features)
        vertex_features = jnn.relu(vertex_features)

        vertex_features = adjacency @ vertex_features
        vertex_features = self.linear2(vertex_features)
        vertex_features = self.norm2(vertex_features)
        vertex_features = jnn.relu(vertex_features)
        return vertex_features + residual


def concat_pairwise(x: jax.Array) -> jax.Array:
    concat = lambda a, b: jnp.concatenate([a, b], axis=-1)
    concat = jax.vmap(concat, in_axes=(0, None))
    concat = jax.vmap(concat, in_axes=(None, 0))
    return concat(x, x)


def sigmoid_inv(x, eps=EPS_DEFAULT):
    x = jnp.clip(x, eps, 1 - eps)
    return jnp.log(x / (1 - x))


class OutputLayer(fj.Module):
    def __init__(self, key, dim_in, dim):
        self.dim_in = dim_in
        self.dim = dim

        key1, key2, key3 = jrandom.split(key, 3)
        self.mlp = Sequential(
            Linear(key1, 3 * dim, dim),
            LayerNorm(dim),
            jnn.relu,
            Linear(key2, dim, dim),
            LayerNorm(dim),
            jnn.relu,
            Linear(key3, dim, 1),
        )

    def __call__(self, adjacency, vertex_features):
        adjacency, vertex_features = remove_virtual_node(
            adjacency,
            vertex_features,
        )

        adjacency = sigmoid_inv(adjacency)[..., None]
        edge_features_vertices = concat_pairwise(vertex_features)

        edge_features = fourier_features(adjacency, self.dim)
        edge_features = jnp.concatenate(
            [edge_features, edge_features_vertices], axis=-1
        )

        edge_features = self.mlp(edge_features)
        edge_features = edge_features[..., 0]

        edge_features = edge_features + edge_features.T
        return set_diagonal(edge_features, 0)


class BinaryEdgesModel(fj.Module):
    def __init__(self, key, nlayer, dim):
        key_input, key_gcn, key_output = jrandom.split(key, 3)
        self.input_layer = InputLayer(key_input, dim)

        self.gcn_layers = fj.ModuleList(
            *[GCNLayer(k, dim, dim) for k in jrandom.split(key_gcn, nlayer)],
        )
        self.output_layer = OutputLayer(key_output, dim, dim)

    def __call__(self, noisy_adjacency, sigma):
        adjacency, vertex_features = self.input_layer(noisy_adjacency, sigma)

        for layer in self.gcn_layers.modules:
            vertex_features = layer(adjacency, vertex_features)

        binary_logits = self.output_layer(adjacency, vertex_features)
        return binary_logits


class Encoder(fj.Module):
    def __init__(self, key, nlayer, dim):
        key_input, key_gcn, key_output = jrandom.split(key, 3)
        self.input_layer = InputLayer(key_input, dim)

        self.gcn_layers = fj.ModuleList(
            *[GCNLayer(k, dim, dim) for k in jrandom.split(key_gcn, nlayer)],
        )
        self.output = Linear(key_output, dim, dim)


def test():
    model = BinaryEdgesModel(jrandom.PRNGKey(0), 1, 128)

    # adjacencies = #jnp.zeros((4, 4))
    adjacencies = jrandom.normal(jrandom.PRNGKey(0), (4, 4))
    adjacencies = (adjacencies + adjacencies.T) / 2**0.5

    sigma = jnp.ones(())

    logits = model(adjacencies, sigma)
    print(logits)


if __name__ == "__main__":
    test()

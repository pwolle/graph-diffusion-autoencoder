import flarejax as fj
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
    i = 2 * jnp.pi * 2 ** i

    return jnp.concatenate(
        [
            jnp.sin(x * i),
            jnp.cos(x * i),
        ],
        axis=-1,
    )


def ratio_encoding(binary_with_noise, sigma, prior=0.5, eps=EPS_DEFAULT):
    p0 = jstats.norm.pdf(binary_with_noise, loc=0, scale=sigma) * prior
    p1 = jstats.norm.pdf(binary_with_noise, loc=1, scale=sigma) * (1 - prior)
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

        key1, key2, key3 = jrandom.split(key, 3)
        self.mlp = Sequential(
            Linear(key1, 2 * dim, dim),
            LayerNorm(dim),
            jnn.relu,
            Linear(key2, dim, dim),
            LayerNorm(dim),
            jnn.relu,
            Linear(key3, dim, dim),
        )

    def __call__(self, noisy_adjacency, sigma):
        prob_adjacency = ratio_encoding(noisy_adjacency, sigma, self.prior)
        prob_adjacency = set_diagonal(noisy_adjacency, 1)

        degree = jnp.sum(prob_adjacency, axis=-1, keepdims=True) - 5
        degree = fourier_features(degree, self.dim)

        sigma_encoded = fourier_features(sigma, self.dim)[None]
        sigma_encoded = jnp.repeat(sigma_encoded, degree.shape[-2], axis=-2)

        vertex_features = jnp.concatenate([degree, sigma_encoded], axis=-1)
        vertex_features = self.mlp(vertex_features)
        return prob_adjacency, vertex_features


class GCNLayer(fj.Module):
    def __init__(self, key, dim_in, dim):
        self.dim_in = dim_in
        self.dim = dim

        self.linear1 = Linear(key, dim_in, dim)
        self.linear2 = Linear(key, dim, dim)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

    def __call__(self, e, v):
        r = v
        v = self.linear1(v)
        v = self.norm1(v)
        v = jnn.relu(v)

        v = e @ v
        v = self.linear2(v)
        v = self.norm2(v)
        v = jnn.relu(v)
        return v + r


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

    def __call__(self, prob_adjacency, vertex_features):
        edges_from_vertex = jnp.tile(
            vertex_features[:None],
            (vertex_features.shape[0], 1, 1),
        )
        edges_to_vertex = jnp.concatenate(
            [
                edges_from_vertex,
                edges_from_vertex.transpose(1, 0, 2),
            ],
            axis=-1,
        )

        edge_features = fourier_features(prob_adjacency[..., None], self.dim)
        edge_features = jnp.concatenate(
            [edge_features, edges_to_vertex],
            axis=-1,
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
        prob_adjacency, vertex_features = self.input_layer(
            noisy_adjacency,
            sigma,
        )

        for layer in self.gcn_layers.modules:
            vertex_features = layer(noisy_adjacency, vertex_features)

        binary_logits = self.output_layer(
            prob_adjacency,
            vertex_features,
        )
        return binary_logits


def symmetric_normal(
    key,
    shape: tuple[int, ...],
):
    assert len(shape) >= 2
    assert shape[-1] == shape[-2]

    x = jrandom.normal(key, shape)
    x = (x + x.transpose(*range(len(shape) - 2), -1, -2)) / 2**0.5
    return x


def uniform_low_discrepancy(key, size: int, minval, maxval):
    step = (maxval - minval) / size
    bins = jnp.linspace(minval, maxval, size, dtype=jnp.float32)

    x = jrandom.uniform(
        key,
        (size,),
        dtype=jnp.float32,
        minval=0,
        maxval=step,
    )
    return bins + x


def random_sigma(key, size: int, minval: float = 1e-1, maxval: float = 1e1):
    x = uniform_low_discrepancy(
        key,
        size,
        jnp.log(minval),
        jnp.log(maxval),
    )
    return jnp.exp(x)


def bce_logits(binary, logits):
    a = -jnn.log_sigmoid(logits) * binary
    c = -jnn.log_sigmoid(-logits) * (1 - binary)
    return a + c


def score_interpolation_loss(key, adjacencies, model):
    assert adjacencies.ndim == 3

    key_noise, key_sigma = jrandom.split(key, 2)
    noise = symmetric_normal(key_noise, adjacencies.shape)

    batch_size = adjacencies.shape[0]
    sigma = random_sigma(key_sigma, batch_size)

    adjacencies_tilde = adjacencies + noise * sigma[..., None, None]
    adjacencies_hat = model(adjacencies_tilde, sigma)

    assert adjacencies_hat.shape == adjacencies.shape

    loss = bce_logits(adjacencies, adjacencies_hat)
    return loss.mean()


def accuracy(binary, logits):
    return (binary == (logits > 0)).mean()


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

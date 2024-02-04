from typing import Self

import flarejax as fj
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats


EPS_DEFAULT = 1e-6


def fourier_features(x, n: int, eps: float = EPS_DEFAULT):
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


def sigmoid_inv(x, eps=EPS_DEFAULT):
    x = jnp.clip(x, eps, 1 - eps)
    return jnp.log(x / (1 - x))


def set_diagonal(x, value):
    assert x.ndim == 2, f"ndim != 2, got shape {x.shape}"
    assert x.shape[0] == x.shape[1]

    n = x.shape[-1]
    r = jnp.arange(n, dtype=jnp.int32)
    x = x.at[r, r].set(value)
    return x


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


class InputLayer(fj.Module):
    def __init__(self: Self, key: jax.Array, dim: int):
        self.dim = dim

        key, key1, key2 = jrandom.split(key, 3)
        self.global_mlp = Sequential(
            Linear(key1, dim, dim),
            jnn.relu,
            LayerNorm(dim),
            Linear(key2, dim, dim),
            jnn.relu,
            LayerNorm(dim),
        )

        key, key1, key2 = jrandom.split(key, 3)
        self.vertex_mlp = Sequential(
            Linear(key1, dim, dim),
            jnn.relu,
            LayerNorm(dim),
            Linear(key2, dim, dim),
            jnn.relu,
            LayerNorm(dim),
        )

        key1, key2 = jrandom.split(key, 2)
        self.edge_mlp = Sequential(
            Linear(key1, dim, dim),
            jnn.relu,
            LayerNorm(dim),
            Linear(key2, dim, dim),
            jnn.relu,
            LayerNorm(dim),
        )

    def __call__(
        self: Self,
        noisy_adjacency: jax.Array,
        sigma: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        # global_features = fourier_features(sigma[..., None], self.dim)
        # global_features = self.global_mlp(global_features)

        prob_adjacency = ratio_encoding(noisy_adjacency, sigma)
        prob_adjacency = set_diagonal(noisy_adjacency, 1)

        degree = jnp.sum(prob_adjacency, axis=-1, keepdims=True)
        degree = degree - 5
        degree = fourier_features(degree, self.dim)
        vertex_features = self.vertex_mlp(degree)

        edge_features = fourier_features(prob_adjacency[..., None], self.dim)
        edge_features = self.edge_mlp(edge_features)

        return edge_features, vertex_features


class MultiHeadBidirectionalAttention(fj.Module):
    def __init__(self: Self, key: jax.Array, dim: int, dim_at: int, num_heads: int):
        self.dim = dim
        self.dim_at = dim_at
        self.num_heads = num_heads

        keyq, keyk, keyv, keyo, keye = jrandom.split(key, 5)
        self.linearq = Linear(keyq, dim, dim_at * num_heads, bias=False)
        self.lineark = Linear(keyk, dim, dim_at * num_heads, bias=False)
        self.linearv = Linear(keyv, dim, dim_at * num_heads, bias=False)

        self.linearo = Linear(keyo, dim_at * num_heads, dim, bias=False)
        self.lineare = Linear(keye, dim, num_heads, bias=False)

    def __call__(self, edge_features, features_edges):
        q = self.linearq(features_edges)
        k = self.lineark(features_edges)
        v = self.linearv(features_edges)

        q = jnp.reshape(q, q.shape[:-1] + (self.num_heads, self.dim_at))
        k = jnp.reshape(k, k.shape[:-1] + (self.num_heads, self.dim_at))
        v = jnp.reshape(v, v.shape[:-1] + (self.num_heads, self.dim_at))

        a = jnp.einsum("qha,kha->hqk", q, k)
        a = a / jnp.sqrt(self.dim_at)

        e = self.lineare(edge_features)
        e = e.transpose((2, 0, 1))
        e = (e + e.swapaxes(-1, -2)) * 2**-0.5

        e = e * jnp.logspace(-2, 2, self.num_heads, base=2)[:, None, None]
        a = a + e
        a = jnn.softmax(a, axis=-1)

        v = jnp.einsum("hqk,kha->qha", a, v)
        v = jnp.reshape(v, v.shape[:-2] + (self.dim_at * self.num_heads,))
        return self.linearo(v)


class GlobalFromVerticies(fj.Module):
    def __init__(self: Self): ...

    def __call__(self, features, features_auxillary):
        # do cross attention to add information from features_auxillary
        # to the features
        ...


class GraphTransformerBlock(fj.Module):
    """
    https://arxiv.org/abs/1711.07553
    """

    def __init__(self: Self): ...

    def __call__(
        self: Self,
        features_edges: jax.Array,
        features_vertex: jax.Array,
        features_global: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # transformer block with MHAttention
        ...


class OutputLayer(fj.Module):
    def __init__(self: Self): ...

    def __call__(
        self: Self,
        features_edges: jax.Array,
        features_vertex: jax.Array,
        features_global: jax.Array,
    ) -> jax.Array: ...


class GraphTransformerBinaryEdges(fj.Module):
    def __init__(self: Self): ...

    def __call__(
        self: Self,
        noisy_adjacency: jax.Array,
        noise_level: jax.Array,
    ) -> jax.Array: ...


class GraphTransformerBinaryEdgesConditional(fj.Module):
    def __init__(self: Self): ...

    def encode(self, adjacency):
        pass

    def __call__(
        self: Self,
        noisy_adjacency: jax.Array,
        noise_level: jax.Array,
        condition_adjacency: jax.Array | None = None,
    ) -> jax.Array: ...


def test():
    dim = 8
    key = jrandom.PRNGKey(0)

    key, subkey = jrandom.split(key, 2)
    input_layer = InputLayer(subkey, dim)

    key, subkey = jrandom.split(key, 2)
    layer = MultiHeadBidirectionalAttention(subkey, dim, 3, 2)

    # adjacencies = #jnp.zeros((4, 4))
    adjacencies = jrandom.normal(jrandom.PRNGKey(0), (5, 5))
    adjacencies = (adjacencies + adjacencies.T) / 2**0.5

    sigma = jnp.ones(())

    edge_features, vertex_features, global_features = input_layer(adjacencies, sigma)
    print(edge_features.shape)
    print(vertex_features.shape)
    print(global_features.shape)

    vertex_features = layer(edge_features, vertex_features)
    print(vertex_features.shape)


if __name__ == "__main__":
    test()

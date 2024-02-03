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


def concat_pairwise(x: jax.Array) -> jax.Array:
    concat = lambda a, b: jnp.concatenate([a, b], axis=-1)
    concat = jax.vmap(concat, in_axes=(0, None))
    concat = jax.vmap(concat, in_axes=(None, 0))
    return concat(x, x)


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


class MLP(fj.Module):
    def __init__(
        self: Self, key: jax.Array, dim_source: int, dim_hidden: int, dim_target: int
    ) -> None:
        key1, key2 = jrandom.split(key)
        self.main = Sequential(
            Linear(key1, dim_source, dim_hidden),
            jnn.relu,
            LayerNorm(dim_hidden),
            Linear(key2, dim_hidden, dim_target),
            jnn.relu,
            LayerNorm(dim_target),
        )

    def __call__(self: Self, x: jax.Array) -> jax.Array:
        return self.main(x)


class InputLayer(fj.Module):
    def __init__(self, key, dim):
        self.dim = dim
        self.mlp = MLP(key, 2 * dim, 2 * dim, dim)

    def __call__(self, noisy_adjacency, sigma):
        adjacency = ratio_encoding(noisy_adjacency, sigma)
        adjacency = set_diagonal(noisy_adjacency, 0.5)

        degree = jnp.sum(adjacency, axis=-1, keepdims=True) - 4.5
        degree = fourier_features(degree, self.dim)

        sigma_encoded = fourier_features(sigma, self.dim)[None]
        sigma_encoded = jnp.repeat(sigma_encoded, degree.shape[-2], axis=-2)

        vertex_features = jnp.concatenate([degree, sigma_encoded], axis=-1)
        vertex_features = self.mlp(vertex_features)
        return adjacency, vertex_features


class Attention(fj.Module):
    def __init__(self: Self, key: jax.Array, dim: int, dim_at: int, num_heads: int):
        self.dim = dim
        self.dim_at = dim_at
        self.num_heads = num_heads

        keyq, keyk, keyv, keyo, keye = jrandom.split(key, 5)
        self.linearq = Linear(keyq, dim, dim_at * num_heads, bias=False)
        self.lineark = Linear(keyk, dim, dim_at * num_heads, bias=False)
        self.linearv = Linear(keyv, dim, dim_at * num_heads, bias=False)

        self.linearo = Linear(keyo, dim_at * num_heads, dim, bias=False)

    def __call__(self, adjacency, vertex_features):
        q = self.linearq(vertex_features)
        k = self.lineark(vertex_features)
        v = self.linearv(vertex_features)

        q = jnp.reshape(q, q.shape[:-1] + (self.num_heads, self.dim_at))
        k = jnp.reshape(k, k.shape[:-1] + (self.num_heads, self.dim_at))
        v = jnp.reshape(v, v.shape[:-1] + (self.num_heads, self.dim_at))

        a = jnp.einsum("qha,kha->hqk", q, k)
        a = a / jnp.sqrt(self.dim_at)

        e = sigmoid_inv(adjacency)[..., None]
        e = e * jnp.logspace(-2, 2, self.num_heads, base=2)
        e = e.transpose((2, 0, 1))
        e = (e + e.swapaxes(-1, -2)) * 2**-0.5

        e = e * jnp.logspace(-2, 2, self.num_heads, base=2)[:, None, None]
        a = a + e
        a = jnn.softmax(a, axis=-1)

        v = jnp.einsum("hqk,kha->qha", a, v)
        v = jnp.reshape(v, v.shape[:-2] + (self.dim_at * self.num_heads,))
        return self.linearo(v)


class Block(fj.Module):
    def __init__(self, key, dim, dim_at, num_heads):
        self.dim = dim
        self.attention = Attention(key, dim, dim_at, num_heads)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(key, dim, 2 * dim, dim)

    def __call__(self, adjacency, vertex_features):
        vertex_features = self.norm1(vertex_features)
        vertex_features = self.attention(adjacency, vertex_features)
        vertex_features = vertex_features + self.norm2(vertex_features)
        vertex_features = self.mlp(vertex_features)
        return vertex_features


class OutputLayer(fj.Module):
    def __init__(self, key, dim_in, dim):
        self.dim_in = dim_in
        self.dim = dim
        self.mlp = Linear(key, dim * 3, 1)

    def __call__(self, adjacency, vertex_features):
        edges_from_vertices = concat_pairwise(vertex_features)

        adjacency = sigmoid_inv(adjacency)[..., None]
        edge_features = fourier_features(adjacency, self.dim)
        edge_features = jnp.concatenate(
            [edge_features, edges_from_vertices],
            axis=-1,
        )

        edge_features = self.mlp(edge_features)
        edge_features = edge_features[..., 0]
        edge_features = edge_features + edge_features.T
        return set_diagonal(edge_features, 0)


class BinaryEdgesModel(fj.Module):
    def __init__(self, key, nlayer, dim, dim_at, num_heads):
        self.nlayer = nlayer
        self.dim = dim
        self.dim_at = dim_at
        self.num_heads = num_heads

        key_input, key_transformer, key_output = jrandom.split(key, 3)
        self.input_layer = InputLayer(key_input, dim)
        self.blocks = fj.ModuleList(
            *[Block(key_transformer, dim, dim_at, num_heads) for _ in range(nlayer)]
        )
        self.output_layer = OutputLayer(key_output, dim, dim)

    def __call__(self, noisy_adjacency, sigma):
        adjacency, vertex_features = self.input_layer(noisy_adjacency, sigma)

        for block in self.blocks.modules:
            vertex_features = block(adjacency, vertex_features)

        return self.output_layer(adjacency, vertex_features)


def test():
    dim = 8
    key = jrandom.PRNGKey(0)

    key, subkey = jrandom.split(key, 2)
    input_layer = InputLayer(subkey, dim)

    key, subkey = jrandom.split(key, 2)
    model = BinaryEdgesModel(subkey, 2, 8, 3, 4)

    # adjacencies = #jnp.zeros((4, 4))
    adjacency = jrandom.normal(jrandom.PRNGKey(0), (5, 5))
    adjacency = (adjacency + adjacency.T) / 2**0.5

    sigma = jnp.ones(())

    vertex_features = model(adjacency, sigma)
    print(vertex_features.shape)
    print(vertex_features)


if __name__ == "__main__":
    test()

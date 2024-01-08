import flarejax as fj

import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats


class Linear(fj.Module):
    def __init__(self, key, dim_in, dim, bias=False):
        self.dim_in = dim_in
        self.dim = dim

        w = jrandom.normal(key, (dim_in, dim), dtype=jnp.float32)
        self.w = fj.Param(w / jnp.sqrt(dim_in))

        if bias:
            b = jnp.zeros((dim,), dtype=jnp.float32)
            self.b = fj.Param(b)
        else:
            self.b = None

    def __call__(self, x):
        x = x @ self.w.data

        if self.b is None:
            return x

        return x + self.b.data


def layer_norm(x, axis=-1):
    mean = jnp.mean(x, axis=axis, keepdims=True)
    var = jnp.var(x, axis=axis, keepdims=True)
    var = jnp.maximum(var, 1e-6)
    inv = lax.rsqrt(var)
    return (x - mean) / inv


class Sequential(fj.ModuleList):
    def __call__(self, x):
        for module in self.modules:
            x = module(x)

        return x


def fourier_features(x, n=32):
    assert n % 2 == 0

    i = jnp.arange(n // 2, dtype=jnp.float32) - n / 4
    i = 2 * jnp.pi * 2**i

    return jnp.concatenate(
        [
            jnp.sin(x * i),
            jnp.cos(x * i),
        ],
        axis=-1,
    )


def ratio_encoding(e, sigma, prior=0.5):
    p0 = jstats.norm.pdf(e, loc=0, scale=sigma) * prior
    p1 = jstats.norm.pdf(e, loc=1, scale=sigma) * (1 - prior)
    return p1 / (p0 + p1)


def set_diagonal(e, value):
    n = e.shape[-1]
    r = jnp.arange(n, dtype=jnp.int32)
    e = e.at[r, r].set(value)
    return e


class InputLayer(fj.Module):
    def __init__(self, key, dim, prior=0.5):
        self.dim = dim
        self.prior = prior

        # self.linear = Linear(key, 3 * dim, dim)
        key1, key2, key3 = jrandom.split(key, 3)
        self.mlp = Sequential(
            Linear(key1, 3 * dim, dim),
            layer_norm,
            jnn.relu,
            Linear(key2, dim, dim),
            layer_norm,
            jnn.relu,
            Linear(key3, dim, dim),
        )

    def __call__(self, e, sigma):
        e = ratio_encoding(e, sigma, self.prior)

        d = jnp.sum(e, axis=-1, keepdims=True)
        d = fourier_features(d, self.dim * 2)

        s = fourier_features(sigma, self.dim)
        s = jnp.repeat(s[None], d.shape[-2], axis=-2)

        v = self.mlp(jnp.concatenate([d, s], axis=-1))

        e = set_diagonal(e, 1)
        return e, v


class GCNLayer(fj.Module):
    def __init__(self, key, dim_in, dim):
        self.dim_in = dim_in
        self.dim = dim

        self.linear = Linear(key, dim_in, dim)

    def __call__(self, e, v):
        v = self.linear(v)
        v = layer_norm(v)
        v = jnn.relu(v)

        v = e @ v
        v = layer_norm(v)
        return v


class OutputLayer(fj.Module):
    def __init__(self, key, dim_in, dim, dim_at):
        self.dim_in = dim_in
        self.dim = dim
        self.dim_at = dim_at

        key, subkey = jrandom.split(key)
        self.proj = Linear(subkey, dim_in, dim_at * dim)

        key1, key2, key3 = jrandom.split(key, 3)
        self.mlp = Sequential(
            Linear(key1, 2 * dim, dim),
            layer_norm,
            jnn.relu,
            Linear(key2, dim, dim),
            layer_norm,
            jnn.relu,
            Linear(key3, dim, 1),
        )

    def __call__(self, e, v):
        v = self.proj(v).reshape(-1, self.dim, self.dim_at)
        a = jnp.einsum("ikh,jkh->ijk", v, v)
        a = a / jnp.sqrt(self.dim_at)

        e = fourier_features(e[..., None], self.dim)
        e = jnp.concatenate([e, a], axis=-1)
        e = self.mlp(e)

        e = (e + e.transpose(1, 0, 2)) / 2**0.5
        e = set_diagonal(e, 0)
        return e


class BinaryEdgesModel(fj.Module):
    def __init__(self, key, nlayer, dim, dim_at):
        key, subkey = jrandom.split(key)
        self.input_layer = InputLayer(subkey, dim)

        self.gcn_layers = fj.ModuleList(
            *[GCNLayer(k, dim, dim) for k in jrandom.split(key, nlayer)],
        )

        self.output_layer = OutputLayer(key, dim, dim, dim_at)

    def __call__(self, e, sigma):
        e, v = self.input_layer(e, sigma)

        for layer in self.gcn_layers.modules:
            v = layer(e, v)

        e = self.output_layer(e, v)
        return e[..., 0]


def main():
    key = jrandom.PRNGKey(0)
    key, subkey = jrandom.split(key)

    e = jrandom.uniform(subkey, (7, 7), dtype=jnp.float32)
    e = e + e.T
    e = (e > 1.3).astype(jnp.float32)

    key, subkey = jrandom.split(key)
    z = jrandom.normal(subkey, e.shape, dtype=jnp.float32)
    z = (z + z.T) / 2**0.5
    e_tilde = e + z

    # print(e_tilde.shape)

    model = BinaryEdgesModel(key, 1, 32, 2)

    e_hat = model(e_tilde, jnp.array([1.0]))
    print(e_hat.shape)


if __name__ == "__main__":
    main()

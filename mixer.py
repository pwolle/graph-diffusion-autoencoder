from typing import Self

import flarejax as fj
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats


EPS_DEFAULT = 1e-6


def fourier_features(
    x: jax.Array,
    n: int,
    eps: float = EPS_DEFAULT,
):
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


def ratio_encoding(
    binary_with_noise: jax.Array,
    sigma: float | jax.Array,
    eps: float = EPS_DEFAULT,
) -> jax.Array:
    p0 = jstats.norm.pdf(binary_with_noise, loc=0, scale=sigma)
    p1 = jstats.norm.pdf(binary_with_noise, loc=1, scale=sigma)
    return (p1 + eps) / (p0 + p1 + eps)


def set_diagonal(x: jax.Array, value: float | jax.Array) -> jax.Array:
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


def symmetrize_features(x: jax.Array) -> jax.Array:
    assert x.ndim == 3
    x = (x + x.swapaxes(-3, -2)) * 2**-0.5
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


def layer_norm(
    x: jax.Array,
    axis: int = -1,
    eps: float = EPS_DEFAULT,
):
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
        key, key1, key2 = jrandom.split(key, 3)
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
    def __init__(self: Self, key: jax.Array, dim: int):
        self.dim = dim
        global_key, vertex_key, edge_key = jrandom.split(key, 3)

        self.bonds_mlp = MLP(edge_key, dim, dim, dim)
        self.atoms_mlp = MLP(vertex_key, dim, dim, dim)
        self.total_mlp = MLP(global_key, dim, dim, dim)

    def __call__(
        self: Self,
        bonds_noisy: jax.Array,
        sigma: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        total_features = fourier_features(sigma[..., None], self.dim)
        total_features = self.total_mlp(total_features)

        prob_adjacency = ratio_encoding(bonds_noisy, sigma)
        prob_adjacency = set_diagonal(bonds_noisy, 1)

        degree = jnp.sum(prob_adjacency, axis=-1, keepdims=True)
        degree = degree - 5
        degree = fourier_features(degree, self.dim)
        atoms_features = self.atoms_mlp(degree)

        bonds_features = fourier_features(prob_adjacency[..., None], self.dim)
        bonds_features = self.bonds_mlp(bonds_features)

        return bonds_features, atoms_features, total_features


class MixBonds(fj.Module):
    def __init__(self: Self, key: jax.Array, dim: int) -> None:
        self.dim = dim
        self.mlp = MLP(key, dim * 4, dim, dim)

    def __call__(
        self,
        bonds_features: jax.Array,
        atoms_features: jax.Array,
        total_features: jax.Array,
    ) -> jax.Array:
        atoms_features = concat_pairwise(atoms_features)
        total_feautres = jnp.broadcast_to(
            total_features,
            atoms_features.shape[:-1] + total_features.shape[-1:],
        )

        x = jnp.concatenate(
            [bonds_features, atoms_features, total_feautres],
            axis=-1,
        )
        y = self.mlp(x)
        return symmetrize_features(y)


def safe_stdd(x, axis: int | tuple[int, ...] = -1, eps=EPS_DEFAULT):
    return jnp.sqrt(jnp.maximum(jnp.var(x, axis=axis), eps))


class MixAtoms(fj.Module):
    def __init__(self, key, dim) -> None:
        self.dim = dim
        self.mlp = MLP(key, dim * 4, dim, dim)

    def __call__(
        self,
        bonds_features: jax.Array,
        atoms_features: jax.Array,
        total_features: jax.Array,
    ) -> jax.Array:
        bonds_features_mean = jnp.mean(bonds_features, axis=-2)
        bonds_features_stdd = safe_stdd(bonds_features, axis=-2)

        total_features = jnp.broadcast_to(
            total_features,
            atoms_features.shape[:-1] + total_features.shape[-1:],
        )

        x = jnp.concatenate(
            [
                bonds_features_mean,
                bonds_features_stdd,
                atoms_features,
                total_features,
            ],
            axis=-1,
        )
        y = self.mlp(x)
        return y


class MixTotal(fj.Module):
    def __init__(self, key, dim):
        self.dim = dim
        self.mlp = MLP(key, dim * 5, dim, dim)

    def __call__(
        self,
        bonds_features: jax.Array,
        atoms_features: jax.Array,
        total_features: jax.Array,
    ) -> jax.Array:
        bonds_features_mean = jnp.mean(bonds_features, axis=(-3, -2))
        bonds_features_stdd = safe_stdd(bonds_features, axis=(-3, -2))

        atoms_features_mean = jnp.mean(atoms_features, axis=-2)
        atoms_features_stdd = safe_stdd(atoms_features, axis=-2)

        x = jnp.concatenate(
            [
                bonds_features_mean,
                bonds_features_stdd,
                atoms_features_mean,
                atoms_features_stdd,
                total_features,
            ],
            axis=-1,
        )
        y = self.mlp(x)
        return y


class MixerBlock(fj.Module):
    def __init__(self, key, dim):
        bonds_key, atoms_key, total_key = jrandom.split(key, 3)
        self.mix_bonds = MixBonds(bonds_key, dim)
        self.mix_atoms = MixAtoms(atoms_key, dim)
        self.mix_total = MixTotal(total_key, dim)

    def __call__(
        self,
        bonds_features: jax.Array,
        atoms_features: jax.Array,
        total_features: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        bonds_features_new = self.mix_bonds(
            bonds_features,
            atoms_features,
            total_features,
        )
        atoms_features_new = self.mix_atoms(
            bonds_features,
            atoms_features,
            total_features,
        )
        total_features_new = self.mix_total(
            bonds_features,
            atoms_features,
            total_features,
        )
        bonds_features = layer_norm(bonds_features + bonds_features_new)
        atoms_features = layer_norm(atoms_features + atoms_features_new)
        total_features = layer_norm(total_features + total_features_new)
        return bonds_features, atoms_features, total_features


class OutputBonds(fj.Module):
    def __init__(self, key, dim: int):
        self.dim = dim

        key_mix, key_out = jrandom.split(key, 2)
        self.mix_bonds = MixBonds(key_mix, dim)
        self.out = Linear(key_out, dim, 1)

    def __call__(
        self,
        bonds_features: jax.Array,
        atoms_features: jax.Array,
        total_features: jax.Array,
    ) -> jax.Array:
        bonds_features = self.mix_bonds(
            bonds_features,
            atoms_features,
            total_features,
        )
        bonds_features = self.out(bonds_features)
        bonds_features = bonds_features[..., 0]
        bonds_features = set_diagonal(bonds_features, 0)
        return bonds_features


class BinaryEdgesModel(fj.Module):
    def __init__(self: Self, key: jax.Array, dim: int, nlayer: int = 1) -> None:
        key_input, key_mixer, key_output = jrandom.split(key, 3)
        self.input_layer = InputLayer(key_input, dim)
        self.mixer_layers = fj.ModuleList(
            *[MixerBlock(k, dim) for k in jrandom.split(key_mixer, nlayer)]
        )
        self.output_layer = OutputBonds(key_output, dim)

    def __call__(self: Self, bonds_noisy: jax.Array, sigma: jax.Array):
        bonds_features, atoms_features, total_features = self.input_layer(
            bonds_noisy,
            sigma,
        )

        for layer in self.mixer_layers.modules:
            bonds_features, atoms_features, total_features = layer(
                bonds_features,
                atoms_features,
                total_features,
            )

        return self.output_layer(
            bonds_features,
            atoms_features,
            total_features,
        )


def main():
    natoms = 5
    dim = 8

    key = jrandom.PRNGKey(0)
    key_adjacency, _, key_model = jrandom.split(key, 3)

    adjacencies = jrandom.normal(key_adjacency, (natoms, natoms))
    adjacencies = (adjacencies + adjacencies.T) * 2**-0.5
    print(adjacencies)

    sigma = jnp.ones(())

    mixer = BinaryEdgesModel(key_model, dim)
    bonds_features = mixer(
        adjacencies,
        sigma,
    )

    print(bonds_features.shape)
    print(bonds_features)


if __name__ == "__main__":
    main()

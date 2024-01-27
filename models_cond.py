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


def layer_norm(x, axis=-1, eps=EPS_DEFAULT):
    mean = jnp.mean(x, axis=axis, keepdims=True)
    var = jnp.var(x, axis=axis, keepdims=True)
    inv = lax.rsqrt(var + eps)
    return (x - mean) * inv


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
    assert x.ndim == 2
    assert x.shape[0] == x.shape[1]

    n = x.shape[-1]
    r = jnp.arange(n, dtype=jnp.int32)
    x = x.at[r, r].set(value)
    return x

def vertex_condition_concat(vertex_features: jnp.array,
                            condition: jnp.array,
                            ) -> jnp.array:
    """
    If we want to concatenate the array vertex_features of dim (batch_size, natoms, nfeatures) with the array condition of shape (batch_size, Nfeatures)
    along the nfeatures dimension, we have to get the condition in shape (batch_size, natoms, Nfeatures). Then the two arrays can be concatenated into
    one array of shape (batch_size, natoms, nfeatures + Nfeatures)-

    Parameters
    ---
    vertex_features: jnp.array
        Jax array representing the existing vertex features from the decoder.

    condition: jnp.array
        Jax array representing the latent space vector from the encoder.
    
    Returns
    ---
    jnp.array
        Array where the features of the condition are concatenated to the features of the vertex_features array for all atoms.
        Array has then shape (batch_size,natoms,nfeatures+Nfeatures)
    """
    
    natoms = vertex_features.shape[0]

    #Shape the condition vector in to the form of (batch_size, natoms, Nfeatures)
    reshaped_condition = jnp.concatenate([condition[None,:] for i in range(natoms)], axis=0)

    #Now create the new vertex_features array of shape (batch_size,natoms,nfeatures+Nfeatures)
    concat_vertex_features = jnp.concatenate([vertex_features,reshaped_condition], axis=-1)

    return concat_vertex_features
    
class InputLayer(fj.Module):
    def __init__(self, key, dim, prior=0.5):
        self.dim = dim
        self.prior = prior

        key1, key2, key3 = jrandom.split(key, 3)
        self.mlp = Sequential(
            Linear(key1, 2 * dim, dim),
            layer_norm,
            jnn.relu,
            Linear(key2, dim, dim),
            layer_norm,
            jnn.relu,
            Linear(key3, dim, dim),
        )

    def __call__(self, noisy_adjacency, sigma=None):
        if sigma != None:
            prob_adjacency = ratio_encoding(noisy_adjacency, sigma, self.prior)
            prob_adjacency = set_diagonal(noisy_adjacency, 1)

        if sigma == None:
            prob_adjacency = noisy_adjacency
        
        d = jnp.sum(prob_adjacency, axis=-1, keepdims=True)
        d = fourier_features(d, self.dim * 2)

        v = self.mlp(d)
        return prob_adjacency, v


class GCNLayer(fj.Module):
    def __init__(self, key, dim_in, dim):
        self.dim_in = dim_in
        self.dim = dim

        self.linear = Linear(key, dim_in, dim)

    def __call__(self, e, v):
        r = v
        v = self.linear(v)
        v = layer_norm(v)
        v = jnn.relu(v)

        v = e @ v
        v = layer_norm(v)
        return v + r



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

    def __call__(self, prob_adjacency, vertex_features):
        vertex_features = self.proj(vertex_features)
        vertex_features = vertex_features.reshape(-1, self.dim, self.dim_at)

        attention = jnp.einsum(
            "ikh,jkh->ijk",
            vertex_features,
            vertex_features,
        )
        attention = attention / jnp.sqrt(self.dim_at)

        edge_features = fourier_features(prob_adjacency[..., None], self.dim)
        edge_features = jnp.concatenate([edge_features, attention], axis=-1)
        edge_features = self.mlp(edge_features)

        edge_features = edge_features + edge_features.transpose(1, 0, 2)
        edge_features = edge_features / 2**0.5
        return edge_features


class Encoder(fj.Module):
    def __init__(self,key, nlayer, dim, dim_at, dim_latent = 1024):
        key,subkey = jrandom.split(key)
        self.input_layer = InputLayer(subkey,dim)

        self.gcn_layers = fj.ModuleList(
            *[GCNLayer(k, dim, dim) for k in jrandom.split(key, nlayer)],
        )
        
        key1, key2, key3 = jrandom.split(key, 3)
        self.mlp = Sequential(
            Linear(key1, dim, 2*dim),
            layer_norm,
            jnn.relu,
            Linear(key2, 2*dim, 4*dim),
            layer_norm,
            jnn.relu,
            Linear(key3, 4*dim, dim_latent),
        )

    def __call__(self, adjacencies):
        adjacencies, vertex_features = self.input_layer(adjacencies, None)

        for layer in self.gcn_layers.modules:
            vertex_features = layer(adjacencies, vertex_features)

        #Mean Pool over number of atoms
        pooled_vertex_features = jnp.mean(vertex_features, axis=0)

        latent_space = self.mlp(pooled_vertex_features)

        return latent_space
    
class BinaryEdgesModel_cond(fj.Module):
    def __init__(self, key, nlayer, dim, dim_at, dim_latent = 1024):
        key, subkey = jrandom.split(key)
        self.input_layer = InputLayer(subkey, dim)

        self.gcn_layers = fj.ModuleList(
            *[GCNLayer(k, dim + dim_latent, dim + dim_latent) for k in jrandom.split(key, nlayer)],
        )

        self.output_layer = OutputLayer(key, dim + dim_latent, dim + dim_latent, dim_at)

    def __call__(self, noisy_adjacency, sigma, condition):
        prob_adjacency, vertex_features = self.input_layer(noisy_adjacency, sigma)        
        vertex_features = vertex_condition_concat(vertex_features,condition)
        
        for layer in self.gcn_layers.modules:
            vertex_features = layer(noisy_adjacency, vertex_features)

        binary_logits = self.output_layer(prob_adjacency, vertex_features)
        binary_logits = binary_logits[..., 0]
        binary_logits = set_diagonal(binary_logits, 0)
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


def random_sigma(key, size: int, minval: float = 1e-2, maxval: float = 1e2):
    x = uniform_low_discrepancy(
        key,
        size,
        jnp.log(minval),
        jnp.log(maxval),
    )
    return jnp.exp(x)


def bce_logits(binary, logits):
    return -jnn.log_sigmoid(logits) * binary - jnn.log_sigmoid(-logits) * (1 - binary)


def score_interpolation_loss_cond(key, adjacencies, model, model_cond, encoder):
    assert adjacencies.ndim == 3

    key_noise, key_sigma = jrandom.split(key, 2)
    noise = symmetric_normal(key_noise, adjacencies.shape)

    batch_size = adjacencies.shape[0]
    sigma = random_sigma(key_sigma, batch_size)

    adjacencies_tilde = adjacencies + noise * sigma[..., None, None]
    adjacencies_hat = model(adjacencies_tilde, sigma) + model_cond(adjacencies_tilde, sigma, encoder(adjacencies))

    assert adjacencies_hat.shape == adjacencies.shape

    loss = bce_logits(adjacencies, adjacencies_hat)
    return loss.mean()


def get_predictions(key, adjacencies, model):
    key_noise, key_sigma = jrandom.split(key, 2)
    noise = symmetric_normal(key_noise, adjacencies.shape)

    batch_size = adjacencies.shape[0]
    sigma = random_sigma(key_sigma, batch_size)

    adjacencies_tilde = adjacencies + noise * sigma[..., None, None]
    adjacencies_hat = model(adjacencies_tilde, sigma)
    return jnn.sigmoid(adjacencies_hat)


def accuracy(binary, logits):
    return (binary == (logits > 0)).mean()

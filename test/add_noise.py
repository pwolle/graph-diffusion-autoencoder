import jax
import jax.numpy as jnp

def adj_noise(adjs, mean, sigma, prng_key,batch = True):
    """Add symmetric normal distributed noise to adjacency matrices
    
    :param adjs: Array of shape (M,N,N), with M being the batch size and N the number of nodes, unbatched --> (N,N)
    :param mean: Mean of the noise
    :param sigma: standard deviation of noise 
    :param prng_key: PRNG Key used as the random key
    :param batch: Boolean to decide if it is a batched input
    """
    
    #Determine number of nodes
    num_nodes = adjs.shape[1]
    
    if batch:
        #Create Additive noise matrix
        noise = jax.random.normal(prng_key,adjs.shape)*sigma/2 + mean
        
        #Create mask to mask out all diagonal element of each adjacency matrix
        mask = jnp.array([jnp.ones((num_nodes,num_nodes)) - jnp.identity(num_nodes) for i in range(len(adjs))])
        
        #Symmetrise noise matrix for each batch
        noise_symm = (noise + jnp.moveaxis(noise,[1,2],[2,1])) * mask
    else:
        #Create Additive noise matrix
        noise = jax.random.normal(prng_key,adjs.shape)*sigma/2 + mean
        
        #Create mask to mask out diagonal elements
        mask = jnp.ones((num_nodes,num_nodes)) - jnp.identity(num_nodes)
        
        #Symmetrise noise matrix
        noise_symm = (noise + noise.T) * mask
        
    #Add noise to adjacency
    noisy_adj = noise_symm + adjs
    
    return noisy_adj

# cuda11/jaxlib-0.4.9+cuda11.cudnn86-cp39-cp39-manylinux2014_x86_64.whl
# cuda11/jaxlib-0.3.8+cuda11.cudnn82-cp39-none-manylinux2014_x86_64.whl
# pip install jaxlib==0.3.25+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# pip install jaxlib==0.4.9+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# pip install jax==0.4.9 -f https://storage.googleapis.com/jax-releases/jax_releases.html

print('testing jax')
import jax
import jax.numpy as jnp
import jax.random as jrand
print("JAX device:", jax.devices())

@jax.jit
def sample_choice(rng):
    r = jrand.choice(rng, 1000, (95,))
    return r

batch_choice = jax.vmap(sample_choice, in_axes=0, out_axes=0)
a = batch_choice(jrand.split(jrand.PRNGKey(12032), 100))
print(a.sum())
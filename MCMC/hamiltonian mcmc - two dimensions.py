#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import tensorflow_probability as tfp


# In[ ]:


import matplotlib.pyplot as plt


# ### info from the tutorial

# In[ ]:


### How to get the mean, std, acceptance - from the tutorial
#     samples, is_accepted = tfp.mcmc.sample_chain(
#         ...
#         trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

#     sample_mean = tf.reduce_mean(samples)
#     sample_stddev = tf.math.reduce_std(samples)
#     is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))


# ### defining a potential function

# In[ ]:


@tf.function
def potential(displacements):
    return displacements**2


# ### plug into HMCMC

# In[ ]:


# Target distribution is proportional to: `exp(-U)`.
@tf.function
def unnormalized_log_prob(x):
    return -1.*potential(x) # this should be the energy function


# In[ ]:


# Initialize the HMC transition kernel.
num_results = int(2e4)
num_burnin_steps = int(2e3)
adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_log_prob,
        num_leapfrog_steps=3,
        step_size=1.),
    num_adaptation_steps=int(num_burnin_steps * 0.8))


# In[ ]:


# Run the chain (with burn-in).
@tf.function
def run_chain():
    # Run the chain (with burn-in).
    samples = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=tf.ones(4),
        kernel=adaptive_hmc,
        return_final_kernel_results=True,
        trace_fn=None)

    return samples


# ### Visualize

# In[ ]:


s = run_chain()


# In[ ]:


X = s.all_states.numpy()


# In[ ]:


import numpy as np


# In[ ]:


R = np.array([
    [0., 0.],
    [3., 3.],
])


# In[ ]:


import seaborn as sns


# In[ ]:


for i in range(R.shape[0]):
    r = (i*2-1)/R.shape[0]
    cmap = sns.cubehelix_palette(light=1, rot=r, as_cmap=True)

    ax = sns.kdeplot(R[i,0]+X[:,i*R.shape[1]], 
                     R[i,1]+X[:,i*R.shape[1]+1], 
                     cmap=cmap, 
                     shade=True, shade_lowest=False)


# In[ ]:





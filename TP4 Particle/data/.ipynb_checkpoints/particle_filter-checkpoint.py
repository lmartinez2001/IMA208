import numpy as np


class ParticleFilter(object):
    """
    Implements a particle filter.
    
    Parameters:
    
    dim_x: int
        Number of state variables for the particle filter. For example, if
        you are tracking the position and velocity of an object in one
        dimension, dim_x would be 2.
    n_particles: int
        Number of particles for the particle filter
    tau: float between 0 and 1
        tau*n_particles is the threshold on the effective sample size under which
        the particles are resampled and their weights reset to 1/N. If 0 we will never
        resample (SIS behaviour). If 1 we will resample at each step (SIR behaviour).
        
    Attributes:
    
    particles: numpy.array(n_particles, dim_x)
        Current state estimates recorded by all particles. Each row
        contains the state of a particle
    weights: numpy.array(n_particles)
        Weights of the particles. The weights sum to 1 over all particles.
    forward: function of signature forward(particles) -> particles
        Function that will be called to perform the predict step of the particle filter,
        whereby the forward state propagation model (including drift and noise) is applied to all particles.
        This returns updated particles.
    likelihood: function of signature likelihood(particles, z) -> likelihoods
        Function that will be called to perform the update step of the particle filter,
        whereby the measurement model and the observation z are used to compute likelihood values
        p(z|x_i) where x_i=particles[i]. The update step uses these likelihood values to update the weights.
        
    The particle filter can be used like this:
    
    PF = ParticleFilter(dim_x, n_particles=..., tau=..., forward=..., likelihood=...)
    self.particles = ... # Initialize particles
    
    while new time step:
        PF.resample()
        PF.predict()
        z = read_measurement(...)
        PF.update(z)
        mean_state = PF.state_expectation()
        
    """
    
    def __init__(self, dim_x, n_particles=100, tau=0.5, forward=None, likelihood=None):
        self.dim_x = dim_x
        self.n_particles = n_particles
        self.tau = tau
        
        if forward is None: # default state propagation model x_k+1 = x_k
            forward = lambda particles: particles 
        if likelihood is None: # default likelihood values ignore measurement z and are all constant
            likelihood = lambda particles, z: np.ones((particles.shape[0],)) 
        
        # Initialize the particles and weights
        self.particles = np.zeros((n_particles, dim_x))
        self.weights = np.ones((n_particles,)) * 1.0/n_particles
        
    def resample(self, tau=None):
        if tau is None:
            tau = self.tau
            
        threshold = tau*self.n_particles
        
        # Compute the effective sample size
        N_eff = 1./np.sum(self.weights**2)
        
        # resample if necessary
        if N_eff < threshold:
            # Hint: you can use numpy.random.choice
            indices = np.random.choice(self.n_particles, size=self.n_particles, replace=True, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones((n_particles,)) * 1.0/n_particles
            
    def predict(self, forward=None):
        """
        Apply the "forward" model (state propagation model) to the particles
        """
        if forward is None:
            forward = self.forward
            
        self.particles = forward(self.particles)
        
    def update(self, z, likelihood=None):
        """
        Update the weight of the particles using the measurement model in likelihood
        """
        if likelihood is None:
            likelihood = self.likelihood
            
        likelihoods = likelihood(self.particles, z) # likelihoods[i] = p(z|x_i) where x_i = particles[i]
        
        # Hint: formula slide 78 in '3. Measure'
        weights = self.weights * likelihoods
        self.weights = weights/np.sum(weights)
        
    def state_expectation(self):
        """
        Computes the empirical average of the particles as an estimate of the mean of the posterior
        distribution of the state given the observations
        
        Returns: numpy.array((dim_x),) 
            A 1D array containing the mean state.
        """
        
        mean = np.mean(self.particles, axis=0)
        return mean
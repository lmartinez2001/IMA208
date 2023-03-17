import numpy as np


class KalmanFilter(object):
    """
    Implements a Kalman filter.
    
    
    Parameters:
    
    dim_x : int
        Number of state variables for the Kalman filter. For example, if
        you are tracking the position and velocity of an object in one
        dimension, dim_x would be 2.
    dim_z : int
        Number of measurement inputs. For example, if the sensor
        provides you with position in one dimension, dim_z would be 1.
        
        
    Attributes:
    
    x : numpy.array(dim_x, 1) (preferred) or numpy.array(dim_x,)
        Current state estimate. Any call to update() or predict() updates
        this variable.
    P : numpy.array(dim_x, dim_x)
        Current state covariance matrix. Any call to update() or predict()
        updates this variable.
    Q : numpy.array(dim_x, dim_x)
        Process noise covariance matrix. 
    R : numpy.array(dim_z, dim_z)
        Measurement noise covariance matrix.
    F : numpy.array(dim_x, dim_x)
        State Transition matrix.
    H : numpy.array(dim_z, dim_x)
        Measurement/observation matrix.
        
        
    You are responsible for initializing x and P to reasonable values (for instance x to 0 and large P), 
    as well as providing Q, R, F, H. 
    
    f = KalmanFilter(dim_x=..., dim_z=...)
    
    f.x = ...
    f.P = ...
    
    f.F = ...
    f.H = ...
    f.Q = ...
    f.R = ...
    
    Then using the filter will look like a succession of predict/update steps:
    
    while new time step:
        f.predict()
        z = read_measurement(...)
        f.update(z)

    """
    
    def __init__(self, dim_x, dim_z):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
            
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # Initialize with zeros or identity when possible
        self.x = np.zeros((dim_x,1))        # state (mean)
        self.P = np.eye(dim_x)        # uncertainty covariance
        self.Q = np.eye(dim_x)        # process uncertainty
        self.F = np.eye(dim_x)        # state transition matrix
        self.H = np.zeros((dim_z,dim_x))        # measurement matrix
        self.R = np.eye(dim_z)        # measurement uncertainty
        
    
    def predict(self, F=None, Q=None):
        """
        Predict next state (a priori) using the Kalman filter state transition equation
        
        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix
        Q : np.array(dim_x, dim_x), or None
            Optional process noise matrix
            
        Returns: 
            Nothing, state self.x and covariance self.P are updated in place 
        """
        
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
            
        # Mean update
        self.x = F @ self.x #####
        
        # Covariance update
        self.P = F @ self.P @ F.T + Q #####
        
        
    def update(self, z, R=None, H=None):
        """
        Update the state (a posteriori) in response to a new measurement z.
        
        z : (dim_z, 1) or (dim_z,)
            measurement for this update.
        R : np.array(dim_z, dim_z), or None
            Optionally provide R to override the measurement noise for this
            one call.
        H : np.array(dim_z, dim_x), or None
            Optionally provide H to override the measurement matrix for this
            one call.
        
        Returns: 
            Nothing, state self.x and covariance self.P are updated in place 
        """
        
        if R is None:
            R = self.R
        if H is None:
            H = self.H
            
        # Innovation
        y = z - H @ self.x #####
        
        # Innovation covariance
        S = H @ self.P @ H.T + R #####
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        # K = self.P @ H.reshape((-1,1)) @ (S.shape[0] == 1 ? 1/S : np.linalg.inv(S)) #####
        
        # New state mean
        self.x = self.x + K @ y #####
        
        # New state covariance (you can use the symmetric formula from slide 50 for stability)
        self.P = (np.eye(self.dim_x) - K @ H) @ self.P #####
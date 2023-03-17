import numpy as np


class RobotSimulation(object):
    def __init__(self, x0=0, velocity=1,
                 measurement_std=0.0,
                 process_std=0.0):
        """ x0 : initial position
            velocity: (+=right, -=left)
            measurement_std: standard deviation in measurement m
            process_std: standard deviation in process (m/s)
        """
        self.x = x0
        self.velocity = velocity
        self.measurement_std = measurement_std
        self.process_std = process_std

    def move(self, dt=1.0):
        """Compute new position of the walker in dt seconds."""
        dx = self.velocity + np.random.randn()*self.process_std
        self.x += dx * dt

    def locate(self):
        """ Returns measurement of new position in meters."""
        measurement = self.x + np.random.randn()*self.measurement_std
        return measurement

    def move_and_locate(self):
        """ Move robot, and return measurement of new position in meters"""
        self.move()
        return self.locate()
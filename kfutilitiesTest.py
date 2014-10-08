#!/usr/bin/python
from kfutilities import *
import unittest

from numpy import matrix, allclose

class TestKfutilities(unittest.TestCase):

    def setUp(self):
        """
        Setup System: 
          Constant-velocity particle
          dx/dt = Fx + Gw
          F = [[0 1],   G = [[0]
               [0 0]]        [1]]
        
          E(w^2) = Q = 1.0
        
        Notice what makes this is a good example is that F * F = 0
        and so computations involving the state transition matrix
        are easy to compute.
        
        Reference:
        Professor Ian Reid's Estimation II Lecture Notes,
        Hilary Term 2001, example on page 12 of the pdf
        http://www.robots.ox.ac.uk/~ian/Teaching/Estimation/LectureNotes2.pdf
        """
        
        self.F = matrix([[0,1],[0,0]])
        self.G = matrix([0,1]).T
        self.Q = 1.0
        
    def get_Q_k_true(self, dt):
        # Returns the true discrete process noise for the system
        # described in the setUp() function
        return self.Q * matrix([[1.0/3.0 * dt**3, 0.5 * dt**2],
                                [0.5*dt**2      ,          dt]])
                                
    def get_Phi_true(self, dt):
        # Returns the true state transition matrix for the system
        # described in the setUp() function
        return matrix([[1, dt],[0, 1]])
    
        
    def test_F2Phi(self):
        dt = 0.1
        
        Phi = F2Phi(self.F, dt)
        self.assertTrue(allclose(Phi, self.get_Phi_true(dt),))
        
    def test_discrete_process_noise(self):
        
        # For small time interval, ensure the default calculation matches
        dt = 0.001                           
        Q_k = discrete_process_noise(self.F, self.G, dt, self.Q)
        self.assertTrue(allclose(Q_k, self.get_Q_k_true(dt)))        
        
        # For larger time intervals, higher order approximations is needed
        dt = 0.01
        Q_k = discrete_process_noise(self.F, self.G, dt, self.Q)
        self.assertFalse(allclose(Q_k, self.get_Q_k_true(dt)))
        
        Q_k = discrete_process_noise(self.F, self.G, dt, self.Q, order = 3)
        self.assertTrue(allclose(Q_k, self.get_Q_k_true(dt)))
        
        Q_k = discrete_process_noise(self.F, self.G, dt, self.Q, order = 5)
        self.assertTrue(allclose(Q_k, self.get_Q_k_true(dt)))
        
        # Alternative scalar example, demonstrating the importance of the
        # approximation order.
        F = matrix(-1); dt = 1.1; Q = matrix(2); G = matrix(1)
        
        # For lower order poor approximations, negative covariance is returned.
        # A higher order approximation can handle this example.
        Q_k = discrete_process_noise(F, G, dt, Q, order = 2)
        self.assertFalse( Q_k > 0.0)
        
        Q_k = discrete_process_noise(F, G, dt, Q, order = 3)
        self.assertTrue( Q_k > 0.0)
        
        # Ensure higher order approximation is close to best available
        # computation for a given example
        dt = 0.05
        Q_k3 = discrete_process_noise(F, G, dt, Q, order = 3)
        Q_k4 = discrete_process_noise(F, G, dt, Q, order = 4)
        Q_k_best  = discrete_process_noise(F, G, dt, Q, order = 5)
        self.assertFalse(allclose(Q_k3, Q_k_best))
        self.assertTrue (allclose(Q_k4, Q_k_best))
        
    def test_cov2corr(self):
        
        # Setup sample covariance and associated correlation matrix
        cov = matrix('1.0  1.0  8.1;\
                    1.0 16.0 18.0;\
                    8.1 18.0 81.0')
                    
        corr = matrix('1.0  0.25 0.9;\
                       0.25 1.0  0.5;\
                       0.9  0.5  1.0')
        
        # Compute correlation using cov2corr() and check answer               
        self.assertTrue(allclose(cov2corr(cov), corr))
                       
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKfutilities)
    unittest.TextTestRunner(verbosity=2).run(suite)

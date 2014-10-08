#!/usr/bin/python
import imu_sensor
import unittest
import StringIO

from numpy import all
import numpy as np

# Additional tests worth adding: test for handling negative timestep

class TestImuSensorClass(unittest.TestCase):

    def setUp(self):
        self.data = 't	p	q	r	ax	ay	az \n\
        0.00	 0.0	 0.0	 -0.00574	 -0.36119	 -1.50244	 0.0\n\
        0.20	 0.0	 0.0	 -0.00573	 -0.36048	 -1.49998	 0.0\n\
        0.40	 0.0	 0.0	 -0.00572	 -0.35511	 -1.49753	 0.0\n\
        0.60	 0.0	 0.0	 -0.00571	 -0.35443	 -1.49507	 0.0\n\
        0.80	 0.0	 0.0	 -0.00571	 -0.35220	 -1.49260	 0.0\n\
        1.00	 0.0	 0.0	 -0.00570	 -0.34997	 -1.49013	 0.0\n\
        1.20	 0.0	 0.0	 -0.00569	 -0.34931	 -1.48765	 0.0\n\
        1.40	 0.0	 0.0	 -0.00568	 -0.34398	 -1.48518	 0.0\n\
        1.60	 0.0	 0.0	 -0.00567	 -0.34334	 -1.48270	 0.0\n\
        1.80	 0.0	 0.0	 -0.00567	 -0.34115	 -1.48021	 0.0\n\
        2.00	 0.0	 0.0	 -0.00566	 -0.33896	 -1.47773	 0.0\n\
        2.20	 0.0	 0.0	 -0.00565	 -0.33834	 -1.47523	 0.0\n\
        2.40	 0.0	 0.0	 -0.00564	 -0.33306	 -1.47274	 0.0\n\
        2.60	 0.0	 0.0	 -0.00563	 -0.33246	 -1.47024	 0.0'
        
    def setup_imu(self):
        lines = StringIO.StringIO(self.data)
        qual = 'consumer'
        
        # Declare two instances of the Imu_sensor class.  One which includes
        # all the defualt errors and the second will all the errors turned off
        self.imu = imu_sensor.Imu_sensor(_file_object = lines,
                                            sensor_qual=qual,
                                                 _store=True)

    def setup_no_error_imu(self):
        lines = StringIO.StringIO(self.data)
        qual = 'consumer'
        
        self.imu_errorsoff = imu_sensor.Imu_sensor(_file_object = lines,
                                              sensor_qual=qual,
                                                   _store=True,
                                                wideband=False,
                                               nullshift=False,
                                               biasdrift=False)

    def setup_only_nullshift_error_imu(self):
        # Returns an imu with only nullshift errors
        
        lines = StringIO.StringIO(self.data)
        qual = 'consumer'
        
        return imu_sensor.Imu_sensor(_file_object = lines,
                                         sensor_qual=qual,
                                             _store=False,
                                           wideband=False,
                                           nullshift=True,
                                          biasdrift=False)    

    def setup_only_biasdrift_error_imu(self):
        lines = StringIO.StringIO(self.data)
        qual = 'consumer'
        
        self.imu_biasdrift_only = imu_sensor.Imu_sensor(_file_object = lines,
                                              sensor_qual=qual,
                                                   _store=True,
                                                wideband=False,
                                               nullshift=False,
                                               biasdrift=True)
    def test_get_imu_true(self):
        # Check that requested true value matches true value.
        self.setup_imu()
        t = 0.0
        data_returned = self.imu.get_imu(t, truth=True)
        data_expected = [0.0, 0.0, -0.00574, -0.36119, -1.50244, 0.0]
        
        # Check each element is near the expected answer.
        for e1, e2 in zip(data_returned, data_expected):
            self.assertAlmostEqual(e1, e2)
        
    def test_get_imu_true_and_errors_off(self):
        # Check for requested value (when errors are off) matches true value.
        self.setup_no_error_imu()
        t = 0.0
        data_returned = self.imu_errorsoff.get_imu(t, truth=True)
        data_expected = [0.0, 0.0, -0.00574, -0.36119, -1.50244, 0.0]
        
        # Check each element is near the expected answer.
        for e1, e2 in zip(data_returned, data_expected):
            self.assertAlmostEqual(e1, e2)

    def test_corrupting_errors(self):
        # Check corrupted measurement differs from uncorrupted output
        self.setup_imu()
        self.setup_no_error_imu()
        t = 1.0
        
        self.assertTrue(all(self.imu_errorsoff.get_imu(t) != self.imu.get_imu(t)))
        
    def test_repeated_calls(self):
        # Check that repeated calls for the same time should return
        # the SAME error values.
        self.setup_imu()
        t = 1.0
        call0 = self.imu.get_imu(0.0)
        call1 = self.imu.get_imu(t)
        call2 = self.imu.get_imu(t)
        
        for e1, e2 in zip(call1, call2):
            self.assertEqual(call1, call2)
            
    def test_varying_timestep(self):
        # Check ability to handle changing time step
        # The Markov process will need to regenerate the discrete-time
        # system to match the new time-step
        #
        # Even though the example data has a uniform time step (0.2 sec),
        # we introduce varying timesteps by calling get_imu() with varying
        # time intervals. 
        self.setup_imu()
        # Ts = 0.2
        call0 = self.imu.get_imu(0.0)
        call1 = self.imu.get_imu(0.2)
        self.assertTrue(self.imu._dt == 0.2)
        
        Qd_g01 = self.imu._Qd_g
        Qd_f01 = self.imu._Qd_f
        
        call2 = self.imu.get_imu(0.5)
        self.assertTrue(self.imu._dt == 0.3)

        Qd_g12 = self.imu._Qd_g
        Qd_f12 = self.imu._Qd_f
        
        # Check that the driving discrete process noise specification has changed
        # and is larger for the larger time step
        self.assertTrue(Qd_g12 > Qd_g01)
        self.assertTrue(Qd_f12 > Qd_f01)
        
    def test_storage(self):
        # Check if N calls to objects causes storage to be of same size
        self.setup_imu()
        
        # Call 4 times
        self.imu.get_imu(0.0)
        self.imu.get_imu(0.2)
        self.imu.get_imu(0.4)
        self.imu.get_imu(0.6)
        
        self.assertTrue(self.imu._ncalls == 4)
        self.assertTrue(len(self.imu._tstore) == 4)
    
    def test_nullshift_generation(self):
        """
        Check that null-shift generator maintains a contstant value for the
        duration of the data run.
        """
        imu_A = self.setup_only_nullshift_error_imu();
        imu_B = self.setup_only_nullshift_error_imu();
        
        t0, t1 = 0.0, 0.2
        
        # Check that null-shift generated is random (i.e. varies from run-to-run)
        call_A0 = imu_A.get_imu(t0)
        call_B0 = imu_B.get_imu(t0)
        
        for e1, e2 in zip(call_A0, call_B0):
            self.assertTrue(e1 != e2)
            
        # Check that null-shift is SAME over time (i.e. constant over a single run)
        call_A1 = imu_A.get_imu(t1)
        
        null_0 = np.array(call_A0) - [0.0, 0.0, -0.00574, -0.36119, -1.50244, 0.0]
        null_1 = np.array(call_A1) - [0.0, 0.0, -0.00573, -0.36048, -1.49998, 0.0]
        self.assertTrue(np.allclose(null_0, null_1))


    def test_biasdrift_statistics(self):
        # Check if specified biasdrift statistics match the computable
        # statistics of the resulting simulated process
        # Note: This test is slow (about 20 sec on laptop) because
        #       it is simulated for 5000 sec in order to test the
        #       statistical properties.  
        
        self.setup_only_biasdrift_error_imu()
        
        # Simulate for Long duration
        # Note: even though the example data only spans several seconds, 
        #       the simulated bias drift will run for the entire requested
        #       time interval.
        
        for t in np.arange(0.0,10000.0, 0.1):
            self.imu_biasdrift_only.get_imu(t)
            
        # Extract realization of x-gyro inrun bias
        b = np.array(self.imu_biasdrift_only._biasdriftstore)[:,0]
        
        # Compute statistics
        mean_expected = 0.0
        std_expected = self.imu_biasdrift_only._sqd['sigma_c_g']
        
        # Check for values to be within two decimal place
        # of the expected value
        print np.std(b)
        self.assertTrue(np.allclose(np.mean(b), mean_expected, atol=0.01))
        self.assertTrue(np.allclose(np.std(b) ,  std_expected, atol=0, rtol=0.2))
 
        
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestImuSensorClass)
    unittest.TextTestRunner(verbosity=2).run(suite)

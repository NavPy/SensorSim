#!/usr/bin/python
import sensor
import unittest
import StringIO

from numpy import isnan, all, any, array
from numpy.linalg import norm
import numbers # to check for scalar number

# Features to add: interpolation bound, number of points

class TestSensorClass(unittest.TestCase):

    def setUp(self):
        data = 't	north	east	down	yaw	pitch	roll \n\
                0.0	3.57	-3.0	0	-2.0	0	0\n\
                0.5	3.57	-3.1	0	-3.0	0	0\n\
                1.0	3.57	-3.2	0	-4.0	0	0\n\
                1.5	3.57	-3.4	0	-5.0	0	0\n\
                2.0	3.57  -3.5	0	-6.0	0	0\n\
                2.5	nan	nan	nan	nan	nan	nan\n\
                3.0	3.57	-3.7	0	-8.0	0	0\n\
                3.5	3.57	-3.8	0	-9.0	0	0\n\
                4.0	3.57	-3.9	0	-10.0	0	0\n\
                4.5	3.57	-4.0	0	-9.0	0	0\n\
                5.0	3.57	-4.1	0	-8.0	0	0'
        lines = StringIO.StringIO(data)
        self.sensor = sensor.Sensor(file_object = lines)

    def test_lastpoint(self):
        """
        Test the last point to still be valid
        """
        data_returned =  self.sensor.get(5.0)
        data_expected = [3.57,	-4.1,	0,	-8.0,	0,	0]
        
        # Check each element is near the expected answer.
        for e1, e2 in zip(data_returned, data_expected):
            self.assertAlmostEqual(e1, e2, places=8)
    
    def test_headerstring(self):
        hdr_string = 't	north	east	down	yaw	pitch	roll'
        self.assertEqual(self.sensor.get_header(),hdr_string)
        
    def test_exact_time(self):
        # make sure the shuffled sequence does not lose any elements
        t1, t2 = 1.5, 4.0
        
        data_returned1 = self.sensor.get(t1)
        data_expected1 = [3.57, -3.4, 0, -5.0 , 0, 0]
        
        data_returned2 = self.sensor.get(t2)
        data_expected2 = [3.57, -3.9, 0, -10.0, 0, 0]
        
        # Check each element is near the expected answer.
        for e1, e2 in zip(data_returned1, data_expected1):
            self.assertAlmostEqual(e1, e2, places=8)

        for e1, e2 in zip(data_returned2, data_expected2):
            self.assertAlmostEqual(e1, e2, places=8)
        
    def test_interpolation(self):
        t = 4.75
        
        data_returned = self.sensor.get(t)       
        data_expected = [3.57, -4.05, 0, -8.375, 0, 0]
        
        # Check each element is near the expected answer.
        for e1, e2 in zip(data_returned, data_expected):
            self.assertAlmostEqual(e1, e2, places=4)
                
    def test_mixed_nan(self):
        t = 2.5
        t_data = self.sensor.get(t)
        self.assertTrue(isnan(t_data).all())
        
    def test_nodata(self):
        t1, t2 = -4, 13
        self.assertTrue(all(isnan(self.sensor.get(t1))))
        self.assertTrue(all(isnan(self.sensor.get(t2))))
        
    def test_rollover(self):
        t1, t2 = 5.0, 0.5

        # Check each element is near the expected answer.
        for e1, e2 in zip(self.sensor.get(t1), [3.57, -4.1, 0, -8.0, 0, 0]):
            self.assertAlmostEqual(e1, e2, places=8)

        for e1, e2 in zip(self.sensor.get(t2), [3.57, -3.1, 0, -3.0, 0, 0]):
            self.assertAlmostEqual(e1, e2, places=8)
        
    def test_min_buffer_size(self):
        t1, t2 = 5.0, 0.0
        
        for e1, e2 in zip(self.sensor.get(t1), [3.57, -4.1, 0, -8.0, 0, 0]):
            self.assertAlmostEqual(e1, e2, places=8)

        for e1, e2 in zip(self.sensor.get(t2), [3.57, -3.0, 0, -2.0, 0, 0]):
            self.assertAlmostEqual(e1, e2, places=8)
    
    def test_dt(self):

        self.assertAlmostEqual(self.sensor.avg_time_step, 0.5)
    
    def test_sep(self):
        """ Check to handle separators other than whitespace."""
        data = 't,	north,	east,	down,	yaw,	pitch,	roll \n\
                0.0, 3.57,	-3.0,	0,	-2.0,	0,	0\n\
                0.5, 3.57,	-3.1,	0,	-3.0,	0,	0\n\
                1.0, 3.57,	-3.2,	0,	-4.0,	0,	0\n\
                1.5, 3.57,	-3.4,	0,	-5.0,	0,	0\n\
                2.0, 3.57,  -3.5,	0,	-6.0,	0,	0\n\
                2.5, nan, nan, nan, nan, nan, nan'
        lines = StringIO.StringIO(data)
        sensor_obj = sensor.Sensor(file_object = lines, sep=',')
        
        t1 = 1.0
        
        for e1, e2 in zip(sensor_obj.get(t1), [3.57, -3.2,	0, -4.0, 0,	0]):
            self.assertAlmostEqual(e1, e2, places=8)       
            
    def test_scalar(self):
        """ Check for scalar data to be returned as scalar (not list)."""
        data = 't,	baroalt\n\
                0.0, 3.57\n\
                0.5, 3.57\n\
                1.0, 3.57\n\
                1.5, 3.57\n\
                2.0, 3.57\n\
                2.5, nan'
        lines = StringIO.StringIO(data)
        sensor_obj = sensor.Sensor(file_object = lines, sep=',')
        
        t1 = 1.0
        h  = 3.57
        h_computed = sensor_obj.get(t1)
        # Check for both scalar type and for accuracy.
        self.assertTrue(isinstance(h_computed, numbers.Number))
        self.assertAlmostEqual(h, h_computed)   
       
    
    # Feature for future implementation: iterate iteratively on data's own rows.    
    #def test_next(self):
    #    self.assertEqual(self.sensor.get(0.0),[3.57, -3.0, 0, -2.0, 0, 0])
    #    self.assertEqual(self.sensor.next(),[3.57, -3.1, 0, -3.0, 0, 0])
    #    self.assertEqual(self.sensor.next(),[3.57, -3.2, 0, -4.0, 0, 0])
    #    self.assertEqual(self.sensor.next(),[3.57, -3.4, 0, -5.0, 0, 0])    


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSensorClass)
    unittest.TextTestRunner(verbosity=2).run(suite)

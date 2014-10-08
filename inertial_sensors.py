import math

# Define constants
cgo = 9.780373 # Constants for a simple gravity model (pp. 53 Titterton)
s2hr = 1.0/3600  # Seconds to hours
d2r  = math.radians(1.0) # Degrees to radians

# TODO (Hamid, 1/8/2013): The units and names for the imu parameters need
#                         to be revisited.  Some may be "root PSD" values.
#                         Also, may decide to create an "input" file used
#                         to specify the sensor quality, which will be loaded
#                         by this class.  This will be similar to the airport
#                         class.

class inertial_sensor():
    """
    Inertial Sensor class defines a set of inertial sensor parameters
    (accelerometers and gyros) for three quality levels:
        1. Navigation Grade
        2. Tactical Grade
        3. Consumer Grade
        
    This was developed originally using the parameters specified in "genImuErr.m"
    written by Professor Demoz Gebre-Egziabher in 2001.
    
    Hamid Mokhtarzadeh
    Last Mod: Jan 8 2013
    """

    def __init__(self):
    
        # Initialize sensor quality dictionary
        sensor_param_dict = {'navigation':{}, 'tactical':{}, 'consumer':{}}
       
        # Generate sensor quality mapping dictionary
        sensor_param_dict['navigation'] = self._generate_navigation_param()
        sensor_param_dict['tactical'] = self._generate_tactical_param()
        sensor_param_dict['consumer'] = self._generate_consumer_param()
        
        # Store ditionary as class property
        self._sensor_param_dict = sensor_param_dict
        
        return
    
    def _generate_navigation_param(self):
        # Navigation Grade
        # Navigation Grade Rate Gyro Error Parameters 
        # Obtained from Ph.D. thesis by Ping Ya Ko titled
        # "GPS-Based Precision Approach and Landing", Stanford University 2000,pp.34
        imu = {}          
        imu['tau_g'] = 3600.0                 #  Time Constant on Gyro Markov Bias [sec]
        imu['sigma_c_g'] = 0.003 * d2r * s2hr #  Standard Deviation of Gyro Markov Bias
        imu['sigma_w_g'] = 0.0008 * d2r       #  Standard Deviation of Gyro Wide Band Noise 
        imu['sigma_n_g'] = imu['sigma_w_g']   #  Null Shift

        imu['tau_f'] = 3600.0                 #  Time Constant on Accelerometer Markov Bias
        imu['sigma_c_f'] = (25.0e-6)*cgo     #  Standard Deviation of Accelerometer Markov Bias
        imu['sigma_w_f'] = (5.0e-6)*cgo      #  Standard Deviation of Accelerometer Wide Band Noise
        imu['sigma_n_f'] = imu['sigma_w_f']   #  Null Shift
        
        return imu
        
    def _generate_tactical_param(self):
        # Tactical Grade Rate Gyro Error Parameters (LN200 Numbers)

        imu = {}            
        imu['tau_g'] = 100.0                 #  Time Constant on Gyro Markov Bias
        imu['sigma_c_g'] = 15.0 * d2r * s2hr #  Standard Deviation of Gyro Markov Bias
# Hamid 4/7/2013: tactical grade imu['sigma_c_g'] was originally 0.35 deg/hr (taken from Demoz).  However the performance was found to be too good.  Therefore for the ITM 2012 simulation work, it was inflated to 15 deg/hr.  Later the quality of the TAC should be revisited to reflect current standards.
        imu['sigma_w_g'] = 0.0017 * d2r      #  Standard Deviation of Gyro Wide Band Noise
        imu['sigma_n_g'] = imu['sigma_w_g']  #  Null Shift

        imu['tau_f'] = 60.0                  #  Time Constant on Accelerometer Markov Bias
        imu['sigma_c_f'] = (50e-6)*cgo      #  Standard Deviation of Accelerometer Markov Bias
        imu['sigma_w_f'] = (50e-5)*cgo      #  Standard Deviation of Accelerometer Wide Band Noise
        imu['sigma_n_f'] = imu['sigma_w_f']  #  Null Shift
        
        return imu
        
    def _generate_consumer_param(self):
        # Automotive Grade Gyro Error Parameters
        
        imu = {}
        imu['tau_g'] = 300.0                  #  Time Constant on Gyro Markov Bias
        imu['sigma_c_g'] = 180.0 * d2r * s2hr #  Standard Deviation of Gyro Markov Bias
        #           TODO: The numbers above and below this line are the same.  Why?
        imu['sigma_w_g'] = 0.05 * d2r         #  Standard Deviation of Gyro Wide Band Noise
        imu['sigma_n_g'] = imu['sigma_w_g']   #  Null Shift

        imu['tau_f'] = 100.0                  #  Time Constant on Accelerometer Markov Bias
        imu['sigma_c_f'] = (1.2e-3)*cgo       #  Standard Deviation of Accelerometer Markov Bias
        imu['sigma_w_f'] = (1.0e-3)*cgo       #  Standard Deviation of Accelerometer Wide Band Noise
        imu['sigma_n_f'] = imu['sigma_w_f']   #  Null Shift
        
        return imu    


    def get_parameters(self, quality):
        """
        Given a sensor quality specification, returns a dictionary of sensor
        error model perameters.
        
        Parameters
        ----------
        quality : string specifying inertial sensor quality.
                  Acceptable values: 'nav' or 'navigation'
                                     'tac' or 'tactical'
                                     'con' or 'consumer'
        
        Returns
        -------
        param_dict : dictionary of sensor quality parameters.
                     The dictionary keys descriptions are as follows:
                     
                         'tau_g': Time Constant on Gyro Markov Bias [sec]
                     'sigma_c_g': Standard Deviation of Gyro Markov Bias
                     'sigma_w_g': Standard Deviation of Gyro Wide Band Noise
                     'sigma_n_g': Null Shift for Gyro [rad/s]
                     
                         'tau_f': Time Constant on Accelerometer Markov Bias [sec]
                     'sigma_c_f': Standard Deviation of Accelerometer Markov Bias
                     'sigma_w_f': Standard Deviation of Accelerometer Wide Band Noise
                     'sigma_n_f': Null Shift for Accelerometer [m/s^2]
    
        """
        if (quality.upper() == 'NAV') or (quality.upper() == 'NAVIGATION'):
            return self._sensor_param_dict['navigation'].copy()
        
        elif (quality.upper() == 'TAC') or (quality.upper() == 'TACTICAL'):
            return self._sensor_param_dict['tactical'].copy()
                    
        elif (quality.upper() == 'CON') or (quality.upper() == 'CONSUMER'):
            return self._sensor_param_dict['consumer'].copy()    
    
    def print_quality(self, quality):
        """
        Prints the parameters for the specified quality level.
        
        Parameters
        ----------
        quality : string specifying inertial sensor quality.
                  Acceptable values: 'nav' or 'navigation'
                                     'tac' or 'tactical'
                                     'con' or 'consumer'
        """
        # Find the parameter dictionary for the desired quality level
        
        imu = self.get_parameters(quality)
        
        print '\n%s Quality \n-------------------------' % quality.upper()
        print ' %30s' % 'GYRO'
        print ' %22s \t %g' % ('Wide Band Noise', imu['sigma_w_g'])
        print ' %22s \t %g' % ('Constant Bias'  , imu['sigma_n_g'])
        print ' %22s \t %g' % ('Bias Stability' , imu['sigma_c_g'])
        print ' %22s \t %g' % ('Bias Correlation Time', imu['tau_g'])
        print '\n'
        
        print ' %35s' % 'ACCELEROMETER'
        print ' %22s \t %g' % ('Wide Band Noise', imu['sigma_w_f'])
        print ' %22s \t %g' % ('Constant Bias'  , imu['sigma_n_f'])
        print ' %22s \t %g' % ('Bias Stability' , imu['sigma_c_f'])
        print ' %22s \t %g' % ('Bias Correlation Time', imu['tau_f'])
        print '\n'
        
if __name__ == '__main__':
    # Example usage of inertial_sensors class
    isensor_obj = inertial_sensor()
    isensor_obj.print_quality('nav')
    isensor_obj.print_quality('tac')
    isensor_obj.print_quality('con')
    

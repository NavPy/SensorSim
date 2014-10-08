import scipy as sp
import numpy as np
import math, sensor
from matplotlib import pyplot as plt
from scipy import signal

from inertial_sensors import inertial_sensor
from kfutilities import discrete_process_noise as disrw

inertial_qual = inertial_sensor()

class Imu_sensor:
    def __init__(self, _file_name = '', _file_object = None,
                                               _store=False, 
                                          sensor_qual='tac',
                                              wideband=True,
                                             nullshift=True,
                                             biasdrift=True):
        """
        Imu_sensor class models a 6 DOF inertial measurement unit (IMU).
        Given a data file with the true imu values, this class will load that
        and dynamically generate the modeled errors as the `measured` imu values
        are requested.  All errors are generated `on-the-fly`, removing
        any requirement to load all the data into memory.
        
        Varying time steps will be handled in two ways:
        1) the true imu values will be interpolated
        2) the errors generated will adapt to the timestep size
        
        Parameters
        ----------
          _file_name: path to file containing true imu values [string]
        _file_object: name of file object open to true imu measurements [file object]
              _store: flag specifying whether values will be stored with time stamps [True/False]
         sensor_qual: string specifying inertial sensor quality.
                  Acceptable values: 'nav' or 'navigation'
                                     'tac' or 'tactical'
                                     'con' or 'consumer'
        
        *Flags specifying what errors will be used to corrupt the imu measurement*
        
         wideband: wideband noise
        nullshift: run-to-run varying bias (constant for duration of run)
                   (note: see self._generate_null_shift_errors documentation)
        biasdrift: bias instability (modeled as first-order Markov process)

        Note: seeding is NOT currently supported
        TODO: The inertial sensor specification is really the "root PSD" and needs to 
               be converted into a STD of noise sample!  You can't use the "root PSD"
               directly!
               This should be updated in parallel with the numbers and units specifications
               of the inertial_sensors object class
        """
    
        # Checks for valid file name or object takes place in Sensor class.
        self._sensor_obj  = sensor.Sensor(file_name = _file_name, file_object = _file_object)
        self.sensor_qual = sensor_qual
        
        # Generate sensor quality dictionary
        self._sqd = inertial_qual.get_parameters(sensor_qual)
        
        # Store flag values
        self._flag_store = _store
        self._flag_wideband  = wideband
        self._flag_nullshift = nullshift
        self._flag_biasdrift = biasdrift
        
        # Initialze a state to count the number of unique time calls
        # to  get_imu() where corrputed measurements were requested.
        self._ncalls = 0

    def _generate_null_shift_errors(self):
        """
        Returns null shift (constant bias) entries in a 6 entry array:
           array([gx_n, gy_n, gz_n, ax_n, ay_n, az_n])
        
        where the subscript 'n' stands for 'null shift'.
        
        Note
        -----
        The constant bias is constant during the run, but varies from run-to-run.  
        
        If, for experimental purposes, you'ld like the SAME constant bias generated
        every run, then this function should be modified to use the standard deviation
        value (not multiplied by random number).
        """
        # If the null-shift value has been generated once, then that value should be used.
        # Otherwise, it will be generated and saved for the next call.        
        try:
            return self._constant_bias
        except AttributeError:          
            # Original code used a non-random null-shift.  This could be acceptable if
            # using a single vehicle, but unrealistic if used for a entire community.  So now the null-shift is random.
            accel_null = self._sqd['sigma_n_f'] * sp.randn(3)
            gyro_null  = self._sqd['sigma_n_g'] * sp.randn(3)
            
            self._constant_bias =  np.hstack((gyro_null, accel_null)) 
            return self._constant_bias
    
    def _generate_wide_band_noise(self):
        """
        Returns a single realization of wide band noise values in a 6 entry array:
            array([gx_w, gy_w, gz_w, ax_w, ay_w, az_w])
            
        where the subscript 'w' stands for 'wide band noise'.        
        """        
        sigma_w_f = self._sqd['sigma_w_f']
        sigma_w_g = self._sqd['sigma_w_g']
        
        noise = np.hstack((sigma_w_g * sp.randn(3), sigma_w_f * sp.randn(3)))
        
        return noise
    
    def _first_order_markov(self, tau, sigma, dt):
        """
        Forms a first order Markov process model of the form:
        dx(t)/dt = (-1/tau) x(t) + w(t)
        
        where w(t) ~ N(0,Qw) is the driving white noise and
        Q = sigma * sigma is the steady state variance of x(t)
        
        Parameters
        ----------
          tau: time constant [sec]
        sigma: steady state (continuous time) standard deviation of x(t)
           dt: discrete system time step
        
        Returns
        -------
            Qw_d: the discrete-time equivalent covariance for the white-noise
                  driving process w(t)
        A_d, B_d: the discrete time system specification for simulating the
                  Markove process: 
                  x(k+1) = A_d * x(k) + B_d * w(k)
                  where w(k) ~ N(0,Qw_d) and A_d, B_d are scalars
        all returned values are scalars      
        """
        
        a = np.matrix(-1.0/tau)
        b, c, d = np.matrix(1.0), np.matrix(1.0), np.matrix(0.0)
        
        # Driving Noise White Power Spectral Density
        # This defines the relationship between the steady state variance
        # and the variance of the driving white noise.
        Qw = np.matrix(2.0 * sigma * sigma / tau)
        
        # Determine the discrete-time equivalent process noise
        # for the driving white process
        Qw_d = disrw(a, b, dt, Qw, order=5)
        
        # Convert continuous time to discrete time system for Markov process
        SS_dis = signal.cont2discrete((a, b, c, d), dt)

        A_d, B_d, C_d, D_d, Ts = SS_dis
        
        return Qw_d.item(), A_d.item(), B_d.item()
    
    
    def _form_biasdrift_model(self, t):
        """
        Builds and stores a discrete time model for the bias drift.
        
        This function may be called repeatedly if the timestep changes
        since the discretization depends on the timestep size.
        
        Parameters
        ----------
        t: time scalar (sec)
        
        Stored values:
           _dt: updated timestep size (sec)
        Ad, Bd: matrices corresponding to state space model for bias drift
                error
                x(k+1) = Ad * x(k) + Bd * u(k)
                where Ad, Bd: 6x6 matrices
                  x(k): 6x1 vector of in-run bias variation error
                        [gx_c, gy_c, gz_c, ax_c, ay_c, az_c]
                        where subscript 'c' corresponds to 'correlated'
                  u(k): 6x1 vector of (discrete) driving white noise
        _Qd_g, _Qd_f: scalars, discrete-time equivalent variance of
                      the white-noise driving process u(k) for the
                      gyro and accelerometer, respectively
        """
        
        # Find time elapsed and generate relevant driving white noise input
        self._dt = t - self._t
        
        # Formulate and store discrete time markov system
        Qd_g, Ad_g, Bd_g = self._first_order_markov(self._sqd['tau_g'],
                                                self._sqd['sigma_c_g'],
                                                self._dt)
        
        Qd_f, Ad_f, Bd_f = self._first_order_markov(self._sqd['tau_f'],
                                                self._sqd['sigma_c_f'],
                                                self._dt)
        
        # Store discrete system state transition matrix 
        # necessary to generate subsequent values
        self._Ad = np.diag([Ad_g, Ad_g, Ad_g, Ad_f, Ad_f, Ad_f])
        
        self._Qd_g = Qd_g
        self._Qd_f = Qd_f
    
    def _generate_markov_bias(self, t):
        """
        Generate realization of bias drift error for current time.
        Returns a 6 entry array:
            array([gx_c, gy_c, gz_c, ax_c, ay_c, az_c])
            
        where the subscript 'c' stands for 'correlated'.
        """
        # this works for an array of 6 values
        
        if self._ncalls == 0:
            # First time called: initialize at zero and return
            self._markov_val = np.array([0.0]*6)
            return self._markov_val
            
        if self._ncalls == 1:
            # Second time called: use second epoch to find "dt" and form discrete system
            self._form_biasdrift_model(t)
            
        elif self._ncalls > 1: 
            # Check "t" to see if the original "dt" is valid.
            # If not, regenerate the discrete-time modle with the new "dt"
            if not np.allclose(self._dt, t - self._t):
                self._form_biasdrift_model(t)
            
        # Use discrete time model to generate driving noise
        ug = sp.sqrt(self._Qd_g) * sp.randn(3)
        uf = sp.sqrt(self._Qd_f) * sp.randn(3)
        u  = np.hstack((ug, uf))

        self._markov_val = np.dot(self._Ad, self._markov_val) + u
        # notice: the _Bd is not used. This is because u is generated to 
        #         match the equivalent distrece statistics of 'Bw(t)' from
        #         the continuous time process.
        #         See Hamid hand notes from 1/10/2013
        
        return self._markov_val

    def get_imu(self, t, truth=False):
        """ 
        Return the vehicle imu measurements at time t [wx,wy,wz,ax,ay,az]
        By default the noisey imu measurements are returned.  If the `truth`
        flag is set to True, then true imu measurements (i.e. no noise) 
        are returned.
        
        Note
        ----
        If the `truth` flag is used, NO errors will be generated and 
        the storage functionality will NOT be called for.
        """
        
        imu_true = self._sensor_obj.get(t) # true imu measurements
        
        # Handle request for true value
        if truth: return imu_true
        
        # Check for repeated calls of same epoch
        if self._ncalls > 0:
            if np.allclose(t, self._t, rtol = 0.0):
                # Note, if rtol is not set to zero, then
                # np.allclose may return true all the timestamps
                # are large in magnitude.
                return self._imu
        
        # Initialize errors matrix, and generate error terms
        # (unless truth is requested)
        wideband  = np.zeros(6)
        nullshift = np.zeros(6)
        biasdrift = np.zeros(6)        

        if self._flag_wideband: 
            wideband += self._generate_wide_band_noise()
            
        if self._flag_nullshift:
            nullshift += self._generate_null_shift_errors()
            
        if self._flag_biasdrift:
            biasdrift += self._generate_markov_bias(t)
                                                   
        # Corrupt truth measurements with sensor errors
        imu = (imu_true + wideband + nullshift + biasdrift).tolist()
        
        # Store values if storage is desired.
        if self._flag_store: self._store(t, imu_true, imu, 
                                        wideband.tolist(),
                                       nullshift.tolist(),
                                       biasdrift.tolist())
        
        # Update state of current time and value
        self._ncalls += 1
        self._t = t
        self._imu = imu[:]
            
        return imu
        

        
    def _store(self, t, imu_true, imu, wideband, nullshift, biasdrift):
        """
        time index, sensor value
        
        Parameters
        ----------
          t: time stamp in numeric format
         imu_true: true imu values [wx, wy, wz, ax, wy, az]
              imu: corrupted imu values
         wideband: wideband errors for the epoch
        nullshift: constant bias error for the epoch
        biasdrift: in-run varying bias for the epoch
        
        All imu_true, wideband, nullshift, and biasdrift parameters
        must be 6 entry lists
        """
        # Define interval for storing full covariance (seconds)
        
        try :
            self._tstore.append(t)
            self._imutruestore.append(imu_true)
            self._imustore.append(imu)            
            self._widebandstore.append(wideband)
            self._nullshiftstore.append(nullshift)
            self._biasdriftstore.append(biasdrift)
            
        except AttributeError:
            # initialize storage
            self._tstore = [t]
            self._imutruestore = [imu_true]
            self._imustore = [imu]
            self._widebandstore = [wideband]
            self._nullshiftstore = [nullshift]
            self._biasdriftstore = [biasdrift]
            
    def plot_sensor(self):
        """
        Generates 6 individual figures, one for each axis.  Each figure has two plots: 
        1) The true IMU values overlaid with the measured values
        2) The errors for that axis
        
        This function will only work if the storage flag is set to true   
        """
        
        if not self._flag_store:
            print 'Unable to plot sensor values since storage flag was ''False'''
            return
        
        titles  = ['X-Gyro', 'Y-Gyro', 'Z-Gyro', 'X-Accel', 'Y-Accel', 'Z-Accel']
        ylabels = ['deg/s' , 'deg/s ', 'deg/s' , 'm/s^2'  , 'm/s^2'  , 'm/s^2'  ]
        units   = [math.degrees(1.0)]*3 + [1.0]*3
        for sensor_ax, title, ylabel, unit in zip(range(6),titles, ylabels, units):
            
            # Extract values of interest
            t = self._tstore
            true_value = np.array(self._imutruestore)[:,sensor_ax]
            value = np.array(self._imustore)[:,sensor_ax]
            
            wideband  = np.array(self._widebandstore)[:,sensor_ax]
            nullshift = np.array(self._nullshiftstore)[:,sensor_ax]    
            biasdrift = np.array(self._biasdriftstore)[:,sensor_ax] 
        
            # Generate respective plot

            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax1.set_ylabel(ylabel)
            ax1.set_title(title)
            
            # Plot measured over true value
            ax1.plot(t, unit * true_value, color='blue', lw=2)
            ax1.plot(t, unit * value, '.', color='blue')

            # Plot errors
            ax2 = fig.add_subplot(212)
            ax2.set_ylabel(ylabel)
            ax2.set_xlabel('Time (sec)')
            
            ax2.plot(t, unit * wideband, label='Wideband')
            ax2.plot(t, unit * nullshift, label='Null Shift')
            ax2.plot(t, unit * biasdrift, label='Bias Drift')
            ax2.legend()
            
        plt.show()
        

        
if __name__ == '__main__':
    # Example usage of IMU Sensor Class
    fobj = open('example_data.txt','r')
    imu = Imu_sensor(_file_object=fobj, sensor_qual='consumer', _store=True)
    
    for t in sp.arange(0,15, 0.1):
        imu.get_imu(t)

    imu.plot_sensor()


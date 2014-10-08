SensorSim
=========

**Note:** This tool has known outstanding issues/limitations and is still being developed and tested.  Once mature, it is planned to migrate into the standard NavPy module as a sub-module.

This module will develop a tool chain to simulate sensors commonly found on navigation systems.  The objective is to define a set of way points and use this module to simulate inertial and magnetometer measurements, as would be measured on board the vehicle.

File Description
----------------

* `inertial_sensors.py` : defines error statistics for three classes of inertial sensors
* `sensor.py` : class that simplifies interaction with time series sensor or trajectory data for navigation simulations.  Associated tests in `sensorTest.py`.
* `imu_sensor.py` : IMU sensor simulator class.  This class uses true IMU values and adds errors based on the sensor quality specification.  Associated tests in `imu_sensorTest.py`.
* `sim_mag.py` : simulate magnetometer measurements as measured on board vehicle.
* `pos2imu.py` : functions for converting 2D position data into *true* imu measurements using backward differencing.
* `kfutilities.py` : utilities for Kalman filtering.  For example, discrete process noise statistics specification.  Associated test in `kfutilitiesTest.py`.

Example: 2D Car
----------------

`traj_gen.py`: define a path for the car using mouse-clicks on map.  Generates the true path, accel, vertical gyro and magnetometer measurements.  This will save several data files.

`sim_car.py`: Load sensor data files, corrupt the true imu output, and dead-reckon forward in time.
import sys
import os
from numpy import array, mod, NaN, mean, diff
from scipy.interpolate import interp1d

# TODO: As part of initialization, rewind the input file_object to be safe
# TODO: when data is replaced with NaN, then the surrounding data points 
#       within the buffer will also get returned as NaN.  This is not ideal.
import functools


# Add momoize to handle repeat calls for the same time's data
# Source: https://gist.github.com/267733
class memoize(object):
    def __init__(self, func):
        self.func = func
        self.memoized = {}
        self.method_cache = {}
    def __call__(self, *args):
        return self.cache_get(self.memoized, args,
            lambda: self.func(*args))
    def __get__(self, obj, objtype):
        return self.cache_get(self.method_cache, obj,
            lambda: self.__class__(functools.partial(self.func, obj)))
    def cache_get(self, cache, key, func):
        try:
            return cache[key]
        except KeyError:
            cache[key] = func()
            return cache[key] 

class Sensor:
    def __init__(self,file_name = '', file_object = None, buffer_size = 4, kind = 'cubic', sep=None):
        """
        
        Parameters:
        -----------
        
        kind: the kind of 1-d interpolation to be used.  Note, anything 
              lower than a `cubic` results in very poor numerical 2nd 
              derivative computations.
              
        sep: specifies delimiter string.  If sep is None, whitespace string is 
             a separator and empty strings are removed.

        """
        # Check for valid constructor inputs
        no_file_name   = len(file_name) == 0
        no_file_object = file_object == None
        
        # Load file or file object
        if no_file_name and no_file_object:
            print 'Invalid input, a valid data file or data object is expected.'
            sys.exit(1)
          
        if not no_file_name:
            if not os.path.isfile(file_name):
                print '\'%s\' is not a valid file location' %name
                sys.exit(1)
                
            self.lines = iter(open(file_name, 'r'))
            
        if not no_file_object:
            self.lines = file_object
        
        # Store sep
        self.sep = sep
            
        # Get header and store number of data columns.  Subtract 1 to remove 
        # time column from count.
        self.header = self.lines.next().strip()
        self.number_of_data_columns = len(self.header.split(self.sep)) - 1
        
        # Load initial buffer.
        self.interp_kind = kind
        self.buffer_size = buffer_size
        self.t_buff = []
        self.data_buff = []
        for line in self.lines:
            t, data = self.load_line(line)
            self.t_buff.append(t)
            self.data_buff.append(data)
            
            if len(self.t_buff) == self.buffer_size: break
            
        # Form data function.
        self.update_fit()
        
        # Store file minimum time.
        self.file_minimum_time = self.t_buff[0]
        
        # Use initial buffer to find approximate time step
        # Note: currently varying time step won't be aputured by the current implementation
        self.avg_time_step = mean(diff(self.t_buff))
        
    def update_fit(self):
        x = self.t_buff
        y = array(self.data_buff).T
        self.data_fit = interp1d(x, y, self.interp_kind)
        
    def get_header(self):
        return self.header
        
    def read_until(self, t_desired):
    
        buff_size = len(self.t_buff)
        buff_center_index = buff_size/2
        if not mod(buff_size,2):
            buff_center_index -= 1
    
        found_flag = False
        for line in self.lines:

            # Free up buffer as necessary
            while len(self.t_buff) >= self.buffer_size:
                self.t_buff = self.t_buff[1:]
                self.data_buff = self.data_buff[1:]
                
            t, data = self.load_line(line)
            self.t_buff.append(t)
            self.data_buff.append(data)
            
            # Set flag as true as soon as desired data comes within buffer.
            # However, if data exists, continue reading until desired point
            # is centered in data buffer AND the buffer is filled.
            if self.t_buff[-1] >= t_desired: found_flag = True
            if (self.t_buff[buff_center_index] >= t_desired) and \
               (len(self.t_buff) == self.buffer_size): break
        
        # Update fit regardless of find status
        self.update_fit()
            
        return True if found_flag else False

    @memoize    
    def get(self,t):
        """
        
        Notes: additional "if t < self.file_minimum_time" statements are to
               return scalars if ony one data column is present.
        """
    
        #ind, val = min(enumerate(self.t_buff), key=lambda x: abs(x[1]-t))
        
        #buff_start = 0
        #buff_end = len(self.t_buff) - 1
        # Check for perfect match
        
        # Try to find data
        try:
            if self.number_of_data_columns > 1:
                return self.data_fit(t).tolist()
            else:
                return self.data_fit(t).tolist()[0]
            
        except ValueError:
            if t < self.file_minimum_time:
                if self.number_of_data_columns > 1:
                    return [NaN] * self.number_of_data_columns            
                else:
                    return NaN
            if t < self.t_buff[0]:
                # Requested value is below buffer, therefore rewind file
                # and reset buffers.
                self.t_buff = []
                self.data_buff = []
                self.lines.seek(0)
                self.lines.next()  # Skip header line.

            if not self.read_until(t):
                if self.number_of_data_columns > 1:
                    return [NaN] * self.number_of_data_columns
                else:
                    return NaN
            else:
                if self.number_of_data_columns > 1:
                    return self.data_fit(t).tolist()
                else:
                    return self.data_fit(t).tolist()[0]

        
    def load_line(self,line):
        """ 
        Given a string of the data file, returns the corresponding 
        (t, data) for that line.  This assumes the first data entry is time.
        
        Blank entries will be converted to NaN.  This is only relevant for
        separators other than whitespaces (e.g. ',')
        """
        
        line_list = line.strip().split(self.sep)
        if self.sep != None:
            # Replace empty string entries with NaN
            line_list = [NaN if x == '' else x for x in line_list]

        float_line = map(float,line_list)
        
        return (float_line[0], float_line[1:])
        
            

import pyfirmata
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.optimize import minimize
from sympy import sin

class HardwareMaster:
    def __init__(self, arduino_usb='COM5', arduino_input=['a:0', 'a:2'], arduino_output='d:6'):
        # USB port
        self.arduino_usb = arduino_usb
        self.arduino_input = arduino_input
        self.arduino_output = arduino_output   
        # Pins for reading and writing
        self.analog_input1 = None
        self.analog_input2 = None
        self.analog_output = None

    def connection_to_arduino(self):
        # Initiate communication with Arduino
        self.board = pyfirmata.Arduino(self.arduino_usb)
        print("Communication successfully started")
        
        # Configure input and output pins
        self.analog_input1 = self.board.get_pin(f'{self.arduino_input[0]}:i')
        self.analog_input2 = self.board.get_pin(f'{self.arduino_input[1]}:i')
        self.analog_output = self.board.get_pin(f'{self.arduino_output}:p')
         
        # Start iterator to receive input data
        it = pyfirmata.util.Iterator(self.board)
        it.start()

    def signal_execution(self, output_signal, Ts):        
        # Execute the given signal
        input_signal_U = []
        input_signal_I = []
        input_time = []
        time_start = time.perf_counter()
        
        for signal in output_signal:
            time_now = time.perf_counter()
            time.sleep(Ts)  # delay for sampling frequency
            print(signal)
            analog_value_U_pre, analog_value_U = self.execute_signal(signal)
            analog_value_I = analog_value_U_pre
            
            input_signal_U.append(analog_value_U)
            input_signal_I.append(analog_value_I)
            input_time.append(time_now - time_start)
        
        return input_signal_U, input_signal_I, input_time

    def execute_signal(self, volt):
        self.analog_output.write(max(0, volt / 256))
        return self.analog_input1.read(), self.analog_input2.read()

class SignalProcessing:
    def __init__(self, plot_index = 0):
        self.plot_index = plot_index
    
    # Carrier frequency filtering
    def bandstop_filter(self, data, init_freq, freq):
        eps = init_freq / freq * 0.1 
        b, a = signal.butter(8, [init_freq / freq - eps, init_freq / freq + eps], 'bandstop')
        return signal.filtfilt(b, a, data)
    
    # Voltage derivative search
    def calculate_derivative(self, U, time):
        dU = [(U[i+1] - U[i]) / (time[i+1] - time[i]) for i in range(len(U) - 1)]
        dU.append(0)
        return np.array(dU)

    def optimize_I_disp(self, C, U, I, time):
        dU = self.calculate_derivative(U, time)
        return np.sum(((dU * C) - np.array(I)) ** 2)
    
    def plot_graph(self, x, y, title=''):
        plt.figure(self.plot_index)
        self.plot_index += 1 
        plt.plot(x, y)
        plt.title(title)
    
    # Noise removal
    def moving_average(self, x, window_size):
        return np.convolve(x, np.ones(window_size), 'valid') / window_size
    # Consolidation of all filters
    def apply_filters(self, I, U, window_size=50, init_freq=50, freqs=100):
        I_filtered = self.bandstop_filter(I, init_freq, freqs)
        U_filtered = self.bandstop_filter(U, init_freq, freqs)
        I_filtered = self.moving_average(I_filtered, window_size)
        U_filtered = self.moving_average(U_filtered, window_size)
        return I_filtered, U_filtered

class DataCollection:
    def __init__(self, polarity='', electrode_system='', amplitude=256, period=2, signal_type=1, full_time=-1, file_path='C:/path/to/data/', file_name='VAX1', file_extension='.csv', R_init=1, Ts=0.01, init_freq=50, use_carrier=0, voltage_coef=1, arduino_max_V=5):
        self.polarity = polarity
        self.electrode_system = electrode_system
        self.amplitude = amplitude
        self.period = period
        self.signal_type = signal_type # signal type: 1-pulse, 2-triangle signal
        self.full_time = full_time
        self.file_path = file_path
        self.file_name = file_name
        self.file_extension = file_extension  
        self.R_init = R_init
        self.use_carrier = use_carrier
        self.init_freq = init_freq # carrier frequency
        self.Ts = Ts
        self.freqs = 1 / self.Ts # sampling frequency
        self.input_signal_U = []
        self.input_signal_I = []
        self.input_time = []
        self.arduino = HardwareMaster()
        self.filtered_signal_I = []
        self.filtered_signal_U = []
        self.rs_I = []
        self.rs_U = []
        self.voltage_coef = voltage_coef
        self.arduino_max_V = arduino_max_V

    def append_to_file(self, item, file_name):
        with open(file_name + self.file_extension, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([item.get("I", ""), item.get("U", ""), item.get("time", "")])
    
    def save_to_file(self, start_record=1):
        for i in range(len(self.input_signal_I)):
            record = {
                'I': self.input_signal_I[i],
                'U': self.input_signal_U[i],
                'time': self.input_time[i]
            }
            if i > start_record:
                self.append_to_file(record, self.file_name)
    
    def read_from_file(self):
        df = pd.read_csv(self.file_path + self.file_name + self.file_extension, delimiter=';', names=['I', 'U', 'time'])
        self.input_signal_U = df['U'].tolist()
        self.input_signal_I = df['I'].tolist()
        self.input_time = df['time'].tolist()

    def create_signal(self, length_module):
        self.output_signal = []
        init_shift = 10
        for i in range(int(2 * self.freqs * length_module)):
            if self.signal_type == 1:  # triangular signal
                # triangle signal
                if init_shift < i < self.freqs * length_module / 2 + init_shift:
                    self.output_signal.append(int(self.amplitude * i / (self.freqs * length_module / 2)) + self.use_carrier * float(sin(self.init_freq * i)))
                elif self.freqs * length_module / 2 <= i < self.freqs * length_module:
                    self.output_signal.append(int(self.amplitude * (2 - 2 * i / (self.freqs * length_module))) + self.use_carrier * float(sin(self.init_freq * i)))
                else:
                    self.output_signal.append(self.use_carrier * float(sin(self.init_freq * i)))
            # pulse signal
            elif self.signal_type == 2: 
                if init_shift < i < self.freqs:
                    self.output_signal.append(256 / 10 + self.use_carrier * float(sin(self.init_freq * i)))
                else:
                    self.output_signal.append(self.use_carrier * float(sin(self.init_freq * i)))
        return self.output_signal

class Experiment:
    def __init__(self, C_inside=0):
        self.C_inside = C_inside
        self.process = SignalProcessing()
        self.arduino = HardwareMaster()
        self.dcvc = None

    def search_parasite_conductivity(self):
        self.dcvc = DataCollection(signal_type=2, file_path='C:/path/to/data/', file_name='VAX1', R_init=1, Ts=0.01, init_freq=50, use_carrier=0, voltage_coef=1e4)
        self.dcvc.create_signal(length_module=self.dcvc.freqs)
        self.arduino.connection_to_arduino()
        self.dcvc.input_signal_U, self.dcvc.input_signal_I, self.dcvc.input_time = self.arduino.signal_execution(self.dcvc.output_signal, self.dcvc.Ts)
        self.dcvc.save_to_file(start_record=4)
        self.process.plot_graph(self.dcvc.input_time, self.dcvc.input_signal_I, 'Input Signal I')
        plt.show()
        self.dcvc.read_from_file()
        self.dcvc.filtered_signal_I, self.dcvc.filtered_signal_U = self.process.apply_filters(self.dcvc.input_signal_I, self.dcvc.input_signal_U, window_size=50, init_freq=self.dcvc.init_freq, freqs=self.dcvc.freqs)
        self.dcvc.filtered_signal_I -= self.dcvc.filtered_signal_I[-1]
        self.dcvc.filtered_signal_U -= self.dcvc.filtered_signal_U[-1]
        res = minimize(self.process.optimize_I_disp, 1e-10, args=(self.dcvc.filtered_signal_U * self.dcvc.arduino_max_V * self.dcvc.voltage_coef, self.dcvc.filtered_signal_I * self.dcvc.arduino_max_V / self.dcvc.R_init, self.dcvc.input_time), method='nelder-mead', options={'xtol': 1e-18, 'disp': True})
        self.C_inside = res.x

    def run_experiment(self):
        self.arduino.connection_to_arduino()
        self.search_parasite_conductivity()
        time.sleep(self.dcvc.Ts * 10)
        self.dcvc = DataCollection(signal_type=1, file_path='C:/path/to/data/', file_name='VAX11', R_init=1, Ts=0.01, init_freq=50, use_carrier=0, voltage_coef=1e4, arduino_max_V=5)
        self.dcvc.create_signal(length_module=4 * self.dcvc.period)
        self.dcvc.input_signal_U, self.dcvc.input_signal_I, self.dcvc.input_time = self.arduino.signal_execution(self.dcvc.output_signal, self.dcvc.Ts)
        self.process.plot_graph(self.dcvc.input_time, self.dcvc.input_signal_I, 'Input Signal I')
        plt.show()
        self.process.plot_graph(self.dcvc.input_time, self.dcvc.input_signal_U, 'Input Signal U')
        plt.show()
        self.dcvc.save_to_file()
        self.dcvc.read_from_file()
        self.dcvc.filtered_signal_I, self.dcvc.filtered_signal_U = self.process.apply_filters(self.dcvc.input_signal_I, self.dcvc.input_signal_U, window_size=50, init_freq=self.dcvc.init_freq, freqs=self.dcvc.freqs)
        self.dcvc.filtered_signal_I -= self.dcvc.filtered_signal_I[-1]
        self.dcvc.filtered_signal_U -= self.dcvc.filtered_signal_U[-1]
        self.rs_I = self.dcvc.filtered_signal_I * self.dcvc.arduino_max_V / self.dcvc.R_init
        self.rs_U = self.dcvc.filtered_signal_U * self.dcvc.arduino_max_V * self.dcvc.voltage_coef
        self.C_inside = 1e-9
        res = minimize(self.process.optimize_I_disp, 1e-10, args=(self.rs_U, self.rs_I, self.dcvc.input_time), method='nelder-mead', options={'xtol': 1e-18, 'disp': True})
        self.C_inside = res.x


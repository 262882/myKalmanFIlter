import numpy as np
import pandas as pd
from typing import Callable
import time

class MyKalmanFilterZeroOrder:
    def __init__(self, xt_init: float, Pt_init: float, R: float, Q: float):
        
        #print("Initialise model")
        
        # State estimates
        self.xt_prev = 0  # previous time period
        self.xt_intr = 0  # intermediate time period
        self.xt_curr = xt_init  # current time period
        
        # State covariances
        self.Pt_prev = 0  # previous time period
        self.Pt_intr = 0  # intermediate time period
        self.Pt_curr = Pt_init  # current time period   
        
        # Noise covariances
        self.r_noise = R  # Measurement noise
        self.q_noise = Q  # System noise
        
        # Kalman gain
        self.k_gain = 0
    
    def _predict(self):
        #print("Predict")
        self.xt_intr = self.xt_prev
        self.Pt_intr = self.Pt_prev + self.q_noise
        
    def _correct(self, yt_measure):
        #print("Update")
        self.k_gain = self.Pt_intr/(self.Pt_intr+self.r_noise)
        self.xt_curr = self.xt_intr + self.k_gain*(yt_measure-self.xt_intr)
        self.Pt_curr = (1-self.k_gain)*self.Pt_intr
    
    def step(self, measurement: float):
        #print("Step")
        self.xt_prev = self.xt_curr
        self.Pt_prev = self.Pt_curr
        self._predict()
        self._correct(measurement)

class MyKalmanFilterHigherOrder:
    def __init__(self, xt_init: np.ndarray, Pt_init: np.ndarray, R: np.ndarray, Q: np.ndarray, F: np.ndarray, H: np.ndarray, B: np.ndarray = np.zeros(1), u: np.ndarray = np.zeros(1)):
        
        #print("Initialise model")
        
        # State estimates
        self.xt_prev = np.empty_like(xt_init)  # previous time period
        self.xt_intr = np.empty_like(xt_init)  # intermediate time period
        self.xt_curr = xt_init                 # current time period
        
        # State covariances
        self.Pt_prev = np.empty_like(Pt_init)  # previous time period
        self.Pt_intr = np.empty_like(Pt_init)  # intermediate time period
        self.Pt_curr = Pt_init                 # current time period   
        
        # Noise covariances
        self.r_noise = R  # Measurement noise
        self.q_noise = Q  # System noise
        
        # Process model
        self.F_trans = F  # State transition matrix
        self.H_measure = H  # Measurement matrix

        # Control model
        self.B_control = B  # Control input matrix
        self.u_input = u  # Control vector
        
        # Kalman gain
        self.k_gain = np.empty_like(Pt_init)
    
    def _predict(self):
        #print("Predict")
        self.xt_intr = self.F_trans@self.xt_prev + self.B_control@self.u_input
        self.Pt_intr = self.F_trans@self.Pt_prev@self.F_trans.T + self.q_noise
        
    def _correct(self, yt_measure):
        #print("Correct")
        self.k_gain = self.Pt_intr@self.H_measure.T/(self.H_measure@self.Pt_intr@self.H_measure.T+self.r_noise)
        self.xt_curr = self.xt_intr + self.k_gain*(yt_measure-self.H_measure@self.xt_intr)
        self.Pt_curr = (1-self.k_gain@self.H_measure)*self.Pt_intr
    
    def step(self, measurement: np.ndarray):
        #print("Step")
        self.xt_prev = np.copy(self.xt_curr)
        self.Pt_prev = np.copy(self.Pt_curr)
        self._predict()
        self._correct(measurement)

class MyKalmanFilterEKF:
    def __init__(self, xt_init: np.ndarray, Pt_init: np.ndarray, R: np.ndarray, Q: np.ndarray, F: np.ndarray, H: np.ndarray,
                  f: Callable, h: Callable, B: np.ndarray = np.zeros(1), u: np.ndarray = np.zeros(1)):
        
        #print("Initialise model")
        
        # State estimates
        self.xt_prev = np.empty_like(xt_init)  # previous time period
        self.xt_intr = np.empty_like(xt_init)  # intermediate time period
        self.xt_curr = xt_init                 # current time period
        
        # State covariances
        self.Pt_prev = np.empty_like(Pt_init)  # previous time period
        self.Pt_intr = np.empty_like(Pt_init)  # intermediate time period
        self.Pt_curr = Pt_init                 # current time period   
        
        # Noise covariances
        self.r_noise = R  # Measurement noise
        self.q_noise = Q  # System noise
        
        # Process model
        self.F_trans = F  # State transition matrix
        self.f_model = f  # Nonlinear function
        self.H_measure = H  # Measurement matrix
        self.h_model = h  # Nonlinear function

        # Control model
        self.B_control = B  # Control input matrix
        self.u_input = u  # Control vector
        
        # Kalman gain
        self.k_gain = np.empty_like(Pt_init)
    
    def _predict(self):
        #print("Predict")
        self.xt_intr = self.f_model(self.xt_prev + self.u_input)
        self.Pt_intr = self.F_trans@self.Pt_prev@self.F_trans.T + self.q_noise
        
    def _correct(self, yt_measure):  
        #print("Correct")
        self.k_gain = self.Pt_intr@self.H_measure.T/(self.H_measure@self.Pt_intr@self.H_measure.T+self.r_noise)
        self.xt_curr = self.xt_intr + self.k_gain*(yt_measure-self.h_model(self.xt_intr))
        self.Pt_curr = (1-self.k_gain@self.H_measure)*self.Pt_intr
    
    def step(self, measurement: np.ndarray):
        #print("Step")
        self.xt_prev = np.copy(self.xt_curr)
        self.Pt_prev = np.copy(self.Pt_curr)
        self._predict()
        self._correct(measurement)

def run_filter(k_filter, measurements, timing=False):
    xt_intr_list = []
    Pt_intr_list = []
    k_gain_list = []
    xt_curr_list = []
    Pt_curr_list = []

    
    time_start = time.perf_counter()

    for cnt, item in enumerate(measurements):
        k_filter.step(item);
        
        xt_intr_list.append(k_filter.xt_intr)
        Pt_intr_list.append(k_filter.Pt_intr)
        k_gain_list.append(k_filter.k_gain)
        xt_curr_list.append(k_filter.xt_curr)
        Pt_curr_list.append(k_filter.Pt_curr)

    if timing == True:
        speed = cnt/(time.perf_counter()-time_start)
        print('Frequency: ', speed, "Per second")
        
    list_of_lists = [xt_intr_list,Pt_intr_list,k_gain_list,xt_curr_list,Pt_curr_list]
    return pd.DataFrame(list(zip(*list_of_lists)), columns= ['xt_intr','Pt_intr','k_gain','xt_curr','Pt_curr'])
import numpy as np

class MyKalmanFilter:
    def __init__(self, xt_init, Pt_init, process_model, measure_noise, system_noise):
        
        print("Initialise model")
        
        # State estimates
        self.xt_prev = np.empty_like(xt_init)  # previous time period
        self.xt_intr = np.empty_like(xt_init)  # intermediate time period
        self.xt_curr = xt_init                 # current time period
        
        # State covariances
        self.Pt_prev = np.empty_like(Pt_init)  # previous time period
        self.Pt_intr = np.empty_like(Pt_init)  # intermediate time period
        self.Pt_curr = Pt_init                 # current time period   
        
        # Noise covariances
        self.r_noise = measure_noise  # Measurement noise
        self.q_noise = system_noise  # System noise
        
        # Process model
        self.model = process_model
        
        # Kalman gain
        self.k_gain = np.empty_like(Pt_init)
    
    def _predict(self):
        print("Predict")
        self.xt_intr = self.model(self.xt_prev)
        self.Pt_intr = self.Pt_prev + self.q_noise
        
    def _update(self, yt_measure):
        print("Update")
        self.k_gain = self.Pt_intr/(self.Pt_intr+self.r_noise)
        self.xt_curr = self.xt_intr + self.k_gain*(yt_measure-self.xt_intr)
        self.Pt_curr = (1-self.k_gain)*self.Pt_intr
    
    def step(self, measurement):
        print("Step")
        self.xt_prev = self.xt_curr
        self.Pt_prev = self.Pt_curr
        self._predict()
        self._update(measurement)
        
        
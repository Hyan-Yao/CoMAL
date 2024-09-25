class IncrementalPID:
    def __init__(self, P:float ,I:float ,D:float, sysInput:float, sysOutput:float):
        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.PIDOutput = sysInput       
        self.SystemOutput = sysOutput    
        self.LastSystemOutput = 0.0      

        self.Error = 0.0
        self.LastError = 0.0
        self.LastLastError = 0.0

    def reset(self, sysInput:float, sysOutput:float):
        self.PIDOutput = sysInput      
        self.SystemOutput = sysOutput  
        self.LastSystemOutput = 0.0    

        self.Error = 0.0
        self.LastError = 0.0
        self.LastLastError = 0.0

    def SetStepSignal(self, StepSignal):
        self.Error = StepSignal - self.SystemOutput

        IncrementalValue = self.Kp*(self.Error - self.LastError)\
            + self.Ki * self.Error +self.Kd *(self.Error -2*self.LastError +self.LastLastError)

        self.PIDOutput += IncrementalValue
        self.LastLastError = self.LastError
        self.LastError = self.Error

        print(StepSignal, self.PIDOutput)

        return self.PIDOutput

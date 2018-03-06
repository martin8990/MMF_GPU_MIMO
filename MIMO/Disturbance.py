import numpy as np
import math
import matplotlib.pyplot as plt
def Crostalk(Signal,NMux,Protection):
    CTMatrix = np.empty((NMux,NMux))
    DistortedSig = np.zeros_like(Signal)
    for m in np.arange(0,NMux):
        B = np.random.dirichlet(np.ones(CrossTalkReject),size=1)
   
        NonCT = np.sum(B[0,0 : CrossTalkReject-NMux])

        CTVector = np.empty(NMux)
        counter = 0
        for l in np.arange(0,NMux):
            if l!=k:
                CTVector[l] = B[0,CrossTalkReject-NMux + counter]
            else:        
                CTVector[l] = NonCT
            counter+=1
        CTVector = CTVector.reshape(1,NMux)
        CTMatrix[:,k] = CTVector

    print("CrossTalk Matrix : ")
    print(CTMatrix)
    
    for k in np.arange(0,N):
    
        FD_Signal[:,k] = np.fft.fft(Signal[:,k])
        FD_Signal[:,k] = np.dot(CTMatrix ,FD_Signal[:,k])
        DistordedSig[:,k] = np.fft.ifft(FD_Signal[:,k]) 
    return 


## DelayConfig
Delay_1 = 1e-10
Delay_2 = 2e-10

T = 1e-6 ## Channel changes every milisecond

pi = math.pi

def DynamicDelay(Signal,f,f_s):
    
    NMux = len(Signal[:,0])
    NSamps = len(Signal[0,:])
    DelayedSig = np.zeros_like(Signal)

    T_osc = T/(1/f_s)
    
    t = (np.sin(np.arange(NSamps)*2*pi * 1/(T_osc))+1)/2


    for Mux in range(NMux):
        X = np.fft.fft(Signal[Mux,:])
        H_1 = np.exp(-1j * Delay_1 * f)
        H_2 = np.exp(-1j * Delay_2 * f)
        H_3 = np.exp(-1j * 1.2e-10 * f)
        Y_1 = X/H_1
        Y_2 = X/H_2
        Y_3 = X/H_3
        
        y_1 = np.fft.ifft(Y_1)
        y_2 = np.fft.ifft(Y_2)
        y_3 = np.fft.ifft(Y_3) 
        #plt.figure()

        DelayedSig[Mux,:] = y_1 * t + (1-t) * y_2 
        #plt.plot(y_1.real,label = "Delay 1")
        #plt.plot(y_2.real,label = "Delay 2")
        #plt.plot(y_3.real,label = "Intermediate Delay")
        #plt.plot(DelayedSig[Mux,:].real,"--",label = "Interpolated")
        #plt.grid(1)
        #plt.xlim(0,10)
        #plt.legend()
       
        
    return DelayedSig

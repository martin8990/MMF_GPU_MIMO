import numpy as np
import matplotlib.pyplot as plt

import Transmitter
import Disturbance
import FDEOvsampleMimo
import math
import MIMO
import All_IN_MIMO

l_b = 64
NSyms = l_b*100
NMux = 1  
Modulation = np.array([1+1j,1-1j,-1+1j,-1-1j]) # QPSK
ovsmpl = 2
f_s = 56e9

mu = 1e-4
#################################### Transmitter ##############################
ShowModulation = True
C_type = np.complex128 

signal = Transmitter.TransmitSignal(NMux,NSyms,ovsmpl,C_type,Modulation)
f = Transmitter.CreateFrequencySpectrum(NSyms*ovsmpl,f_s)

## Distortions ##
signal_Disturbed = Disturbance.DynamicDelay(signal,f,f_s)

signal_e1 = MIMO.FDE_MIMO_GPU(signal_Disturbed,l_b,mu,ovsmpl)
signal_e2 = All_IN_MIMO.ALL_IN_MIMO(signal_Disturbed,l_b,mu,ovsmpl)

plt.figure()
plt.subplot(2,1,1)
diff = np.abs(signal_e1[0,:])-np.abs(signal_e2[0,:])
plt.plot(diff)
plt.subplot(2,1,2)
r = range(int(NSyms/2),NSyms - l_b*2)
plt.scatter(signal_Disturbed[0,r].real,signal_Disturbed[0,r].imag)
plt.scatter(signal_e2[0,r].real,signal_e2[0,r].imag)

plt.show()




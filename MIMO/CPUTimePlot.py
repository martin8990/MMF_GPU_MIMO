
import numpy as np
import MIMO
import numba.cuda as cuda
from timeit import default_timer as timer
import Transmitter
import matplotlib.pyplot as plt  
import MIMO_CPU
import All_IN_MIMO

#plt.style.use('IEEE.mplstyle')

mu = 1e-4
L_b = 128
NMux = range(1,12)

NSyms = int(5000)
ovsmpl = [2]
C_type = np.complex128
Modulation = np.array([1+1j,1-1j,-1+1j,-1-1j]) # QPSK

NS = len(ovsmpl)
NM = len(NMux)


times = np.zeros(NS*NM).reshape(NS,NM)


for S in range(NS):
            
        
    for MUX in range(NM):
       signal = Transmitter.TransmitSignal(NMux[MUX],NSyms,ovsmpl[S],C_type,Modulation)
            
       print("Now Executing With:")
       print("NSyms " + str(NSyms))
       print("OvSample " + str(ovsmpl[S]))
       print("NMUX " + str(NMux[MUX]))
      
       start = timer()

       times[S,MUX] = np.sum(MIMO_CPU.FDE_MIMO_Speedtest(signal,L_b,mu,ovsmpl[S]))/L_b*1e6



plt.figure()
for S in range(NS):
    plt.figure()

    plt.plot(NMux,times[S,:])
    
    plt.xlabel("SDM Channels [-]")
    plt.ylabel("Computation time [Sec/MSymbol]")
    plt.grid(1)
    plt.legend()

plt.show()


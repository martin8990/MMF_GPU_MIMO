
import numpy as np
import MIMO
import numba.cuda as cuda
from timeit import default_timer as timer
import Transmitter
import matplotlib.pyplot as plt  
import MIMO_CPU
import All_IN_MIMO
import matplotlib.gridspec as gridspec

plt.style.use('IEEE.mplstyle')

mu = 1e-4
L_b = range(32,544,32)
NMux = [2,4,6,8,10]

NSyms = int(3000)
ovsmpl = [1]
C_type = np.complex128
Modulation = np.array([1+1j,1-1j,-1+1j,-1-1j]) # QPSK

NS = len(ovsmpl)
NM = len(NMux)
NLB = len(L_b)

times = np.zeros(NS*NM*NLB).reshape(NS,NM,NLB)
times_GPU = np.zeros(NS*NLB).reshape(NS,NLB) # NMUX not relevant :)


for S in range(NS):
    signalGPU = Transmitter.TransmitSignal(NMux[0],NSyms,ovsmpl[S],C_type,Modulation)
    
    for LB in range(NLB):
        times_GPU[S,LB] = np.sum(All_IN_MIMO.ALL_IN_MIMO_SpeedTest(signalGPU,L_b[LB],mu,ovsmpl[S]))
            
        
    for MUX in range(NM):
       signal = Transmitter.TransmitSignal(NMux[MUX],NSyms,ovsmpl[S],C_type,Modulation)
       for LB in range(NLB):
            
            print("Now Executing With:")
            print("NSyms " + str(NSyms))
            print("OvSample " + str(ovsmpl[S]))
            print("NMUX " + str(NMux[MUX]))
            print("L_b " + str(L_b[LB]))
            start = timer()
            MIMO_CPU.FDE_MIMO(signal,L_b[LB],mu,ovsmpl[S])
            cpuTime = timer()-start 
            times[S,MUX,LB] = cpuTime/times_GPU[S,LB]

markertypes = ['-x',"-o","-*","-^","-v","->","-<","-s" ]                     
plt.figure()
for S in range(NS):
    plt.figure()
    gs = gridspec.GridSpec(1, 11)



    plt.subplot(gs[:,0:8])

    for MUX in range(NM):
        plt.plot(L_b,times[S,MUX,:],markertypes[MUX], label = "CH : " + str(NMux[MUX]))
    
    plt.xlabel("BlockLength [-]")
    plt.ylabel("GPU Speedup [-]")
    plt.grid(1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()

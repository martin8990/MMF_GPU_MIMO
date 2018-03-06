
import numpy as np
import MIMO
import numba.cuda as cuda
from timeit import default_timer as timer
import Transmitter
import matplotlib.pyplot as plt  
import MIMO_CPU
import All_IN_MIMO

mu = 1e-4
L_b = [32,64,128,256]
NMux = [2,4,6,8,10]

NSyms = int(3000)
ovsmpl = [1,2,4]
C_type = np.complex128
Modulation = np.array([1+1j,1-1j,-1+1j,-1-1j]) # QPSK

NS = len(ovsmpl)
NM = len(NMux)
NLB = len(L_b)

times = np.zeros(NS*NM*NLB).reshape(NS,NM,NLB)
times_GPU = np.zeros(NS*NLB).reshape(NS,NLB) # NMUX not relevant :)


for S in range(NS):
    for MUX in range(NM):
        signal = Transmitter.TransmitSignal(NMux[MUX],NSyms,ovsmpl[S],C_type,Modulation)
        for LB in range(NLB):
            
            print("Now Executing With:")
            print("NSyms " + str(NSyms))
            print("OvSample " + str(ovsmpl[S]))
            print("NMUX " + str(NMux[MUX]))
            print("L_b " + str(L_b[LB]))

            times[S,MUX,LB] = np.sum(MIMO_CPU.FDE_MIMO_Speedtest(signal,L_b[LB],mu,ovsmpl[S]))/L_b[LB]*1000
            times_GPU[S,LB] = np.sum(All_IN_MIMO.ALL_IN_MIMO_SpeedTest(signal,L_b[LB],mu,ovsmpl[S]))/L_b[LB] * 1000
                    
plt.figure()
for S in range(NS):
    plt.subplot(NS,1,S + 1)
    plt.title("OVSMPL : " + str(ovsmpl[S]))
    plt.plot(L_b,times_GPU[S,:],"-o", label = "GPU, ANY")
    
    for MUX in range(NM):
        plt.plot(L_b,times[S,MUX,:],"-x", label = "CPU : " + str(NMux[MUX]))
    
    plt.xlabel("BlockLength [-]")
    plt.ylabel("Computation Time [sec/kSym]")
    plt.grid(1)
    plt.legend()

plt.show()
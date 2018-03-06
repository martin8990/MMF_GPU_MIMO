import numpy as np
import numba.cuda as cuda
from timeit import default_timer as timer
import Transmitter
import matplotlib.pyplot as plt  

plt.style.use('IEEE.mplstyle')
import All_IN_MIMO
mu = 1e-4
L_b = [256]
NMux = [2, 18]


NSyms = int(3000)
ovsmpl = [2]
C_type = np.complex128
Modulation = np.array([1+1j,1-1j,-1+1j,-1-1j]) # QPSK

NS = len(ovsmpl)
NM = len(NMux)
NLB = len(L_b)

times = np.zeros(NS*NM*NLB*9).reshape(NS,NM,NLB,9)

for S in range(NS):
    for MUX in range(NM):
        signal = Transmitter.TransmitSignal(NMux[MUX],NSyms,ovsmpl[S],C_type,Modulation)
        for LB in range(NLB):
            
            print("Now Executing With:")
            print("NSyms " + str(NSyms))
            print("OvSample " + str(ovsmpl[S]))
            print("NMUX " + str(NMux[MUX]))
            print("L_b " + str(L_b[LB]))

            times[S,MUX,LB,:] = All_IN_MIMO.ALL_IN_MIMO_SpeedTest(signal,L_b[LB],mu,ovsmpl[S])
                
labels = ["Comp(+ 2SUB)","IFFT(1)","(+ SUB) Error","FFT(1)","Error Comp","IFFT(2)","Overlap","FFT(2)","Update"]
labels_CPU = ["Compensate","Error Calculation","Gradient Constraint"]

plt.figure()

mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)
   
width = 0.35
ind = np.arange(NM)
bot = np.zeros(NM)
for t in range(len(times[0,0,0,:])):

    ax.bar(ind,times[0,:,0,t],width = width,label = labels[t],bottom = bot)
    bot += (times[0,:,0,t])
    
plt.grid()
plt.legend()
ax.set_xticks(ind)
ax.set_xticklabels(NMux)
plt.xlabel("SDM channels [A.u]")
plt.ylabel("Computation Time [sec/kSym]")

plt.show()

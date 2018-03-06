import numpy as np
import MIMO
import numba.cuda as cuda
from timeit import default_timer as timer
import Transmitter
import matplotlib.pyplot as plt  
import MIMO_CPU
import All_IN_MIMO
plt.style.use('IEEE.mplstyle')

mu = 1e-4
L_b = [128]
NMux = [2,4]

NSyms = int(3000)
ovsmpl = [2]
C_type = np.complex128
Modulation = np.array([1+1j,1-1j,-1+1j,-1-1j]) # QPSK

NS = len(ovsmpl)
NM = len(NMux)
NLB = len(L_b)

times = np.zeros(NS*(NM+1)*NLB*3).reshape(NS,NM+1,NLB,3)

for S in range(NS):
    for MUX in range(NM):
        signal = Transmitter.TransmitSignal(NMux[MUX],NSyms,ovsmpl[S],C_type,Modulation)
        for LB in range(NLB):
            
            print("Now Executing With:")
            print("NSyms " + str(NSyms))
            print("OvSample " + str(ovsmpl[S]))
            print("NMUX " + str(NMux[MUX]))
            print("L_b " + str(L_b[LB]))

            times[S,MUX+1,LB,:] = MIMO_CPU.FDE_MIMO_Speedtest(signal,L_b[LB],mu,ovsmpl[S])/L_b[0]*1000

    times_GPU = All_IN_MIMO.ALL_IN_MIMO_SpeedTest(signal,L_b[LB],mu,ovsmpl[S])/L_b[0]*1000
    times[S,0,LB,0] = times_GPU[0] + times_GPU[1] + 0.5 * times_GPU[3]
    times[S,0,LB,1] = 0.5 * times_GPU[3] + times_GPU[4] + times_GPU[5] + times_GPU[8]
    times[S,0,LB,2] = times_GPU[6] + times_GPU[7]
                
labels = ["Compensate","Error Calculation","Gradient Constraint"]

plt.figure()

mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)
   
width = 0.35

 
ind = range(NM + 1)
bot = np.zeros(NM+1)

for t in range(len(times[0,0,0,:])):
    
    ax.bar(ind,times[0,:,0,t],width = width,label = labels[t],bottom = bot)
    bot += (times[0,:,0,t])

bot = np.zeros(NM)

tlabels = ["" for x in range(len(ind))]

tlabels[0] = "GPU : ANY"
for k in range(1,len(ind)):
    tlabels[k] = "CPU : " + str(NMux[k-1])


   
plt.grid()
plt.legend()
ax.set_xticks(ind)
ax.set_xticklabels(tlabels)
plt.xlabel("SDM channels [A.u]")
plt.ylabel("Computation Time [sec/kSym]")
plt.show()


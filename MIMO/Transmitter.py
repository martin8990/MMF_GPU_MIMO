import numpy as np
import matplotlib.pyplot as plt
from random import *


def TransmitSignal(NMux,NSyms,ovsmpl,C_type,Modulation):
    Signal = np.zeros(NSyms*NMux*ovsmpl,dtype = C_type).reshape(NMux,NSyms*ovsmpl)
    for m in range(NMux):
        for k in range(NSyms):
            mod =Modulation[randint(0,len(Modulation)-1)]
            Signal[m,k*ovsmpl] = mod
            
            if k>0:
                for s in range(-ovsmpl+1,0):
                    Signal[m,k*ovsmpl + s] += (ovsmpl + s)/ovsmpl*mod
            if k<NSyms-1:
                for s in range(1,ovsmpl):
                    Signal[m,k*ovsmpl + s] += (ovsmpl - s)/ovsmpl*mod

        
    return Signal

def ShowModulation(sig):
    plt.figure()
    plt.subplot(2,1,1)
    for m in range(len(sig[:,0])):
        plt.plot(sig[m,:].real)
    plt.subplot(2,1,2)
    for m in range(len(sig[:,0])):
        plt.plot(sig[m,:].imag)
  


def CreateFrequencySpectrum(NSamps,f_s):
    N = NSamps
    f_pos = np.arange(0,N/2,1)
    f_neg = np.arange(N/2,0 ,-1)
    f = np.concatenate([f_pos,f_neg])
    return f * f_s/N



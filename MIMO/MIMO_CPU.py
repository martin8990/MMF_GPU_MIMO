import numpy as np
import pyculib.fft as GFFT
import numba.cuda as cuda
from MIMOKernels import *
from timeit import default_timer as timer


def FDE_MIMO(x,l_b,mu,ovsmpl,R2 = 2):

    # Preparations
    C_Type = x.dtype
    NSyms =int(len(x[0,:])/ovsmpl)
    NMux = len(x[:,0])

    x_o = np.zeros(NMux*NSyms,dtype = C_Type).reshape(NMux,NSyms)
    
    FirstBlock = range(0,l_b)       # First N Elements
    SecondBlock = range(l_b,2 * l_b)  # Second N Elements
    NBlocks = int(NSyms/l_b - 1)

    O = np.zeros(l_b,dtype = C_Type)
    I = np.ones(l_b,dtype = C_Type) * R2 

    # Setup Central Taps
    h = np.zeros(l_b * 2 * NMux * NMux,dtype = C_Type).reshape(NMux,NMux,l_b * 2)
    H = np.zeros(l_b * 2 * NMux * NMux * ovsmpl,dtype = C_Type).reshape(NMux,NMux,ovsmpl,l_b * 2)
    X_i = np.zeros(NMux*ovsmpl*l_b*2,dtype = C_Type).reshape(NMux,ovsmpl,l_b*2)
    
    CTap = np.int(l_b / 2)
    for OUT in range(NMux):
        h[OUT,OUT,CTap] = 1/ovsmpl + 0j
    for IN in range(NMux):
        for OUT in range(NMux):
            for S in range(ovsmpl):
                H[IN,OUT,S,:] = np.fft.fft(h[IN,OUT,:])
    # Start
    for BLOCK in range(1,NBlocks):


        B_IO = range(BLOCK * l_b - l_b, BLOCK * l_b + l_b) 
        B_O = range(BLOCK * l_b - l_b,BLOCK * l_b)
        
        
        # Compensation
        for IN in range(NMux):
            for OUT in range(NMux):        
                X_k = np.zeros(l_b*2,dtype = np.complex128)
                for S in range(ovsmpl):
                    B_IOS = range(BLOCK * l_b * ovsmpl - l_b * ovsmpl + S , BLOCK * l_b * ovsmpl + l_b * ovsmpl + S,ovsmpl) 
                    X_i[IN,S,:] = np.fft.fft(x[IN,B_IOS])
                   
                X_k = X_k + X_i[IN,S,:] * H[IN,OUT,S,:]           


            x_o[OUT,B_O] = x_o[OUT,B_O] + (np.fft.ifft(X_k)[SecondBlock])
        
        # Tap updates        
        for IN in range(NMux):
            for OUT in range(NMux):        
                for S in range(ovsmpl):
                    
                    e = (I - x_o[OUT,B_O] * np.conj(x_o[OUT,B_O])) * x_o[OUT,B_O]
                    E = np.fft.fft(np.append(O,e)) 
                    s_ = np.fft.ifft(np.conj(X_i[IN,S,:]) * E)[FirstBlock]
                    H[IN,OUT,S,:] = H[IN,OUT,S,:] + mu * np.fft.fft(np.append(s_,O))
    return x_o


def FDE_MIMO_Speedtest(x,l_b,mu,ovsmpl,R2 = 2):

    # Preparations
    C_Type = x.dtype
    NSyms =int(len(x[0,:])/ovsmpl)
    NMux = len(x[:,0])

    x_o = np.zeros(NMux*NSyms,dtype = C_Type).reshape(NMux,NSyms)
    
    FirstBlock = range(0,l_b)       # First N Elements
    SecondBlock = range(l_b,2 * l_b)  # Second N Elements
    NBlocks = int(NSyms/l_b - 1)

    O = np.zeros(l_b,dtype = C_Type)
    I = np.ones(l_b,dtype = C_Type) * R2 

    # Setup Central Taps
    h = np.zeros(l_b * 2 * NMux * NMux,dtype = C_Type).reshape(NMux,NMux,l_b * 2)
    H = np.zeros(l_b * 2 * NMux * NMux * ovsmpl,dtype = C_Type).reshape(NMux,NMux,ovsmpl,l_b * 2)
    X_i = np.zeros(NMux*ovsmpl*l_b*2,dtype = C_Type).reshape(NMux,ovsmpl,l_b*2)
    
    CTap = np.int(l_b / 2)
    for OUT in range(NMux):
        h[OUT,OUT,CTap] = 1/ovsmpl + 0j
    for IN in range(NMux):
        for OUT in range(NMux):
            for S in range(ovsmpl):
                H[IN,OUT,S,:] = np.fft.fft(h[IN,OUT,:])
    # Start
    times =  np.ones(3) * 1e9
    for BLOCK in range(1,NBlocks):

        start = timer()
        B_IO = range(BLOCK * l_b - l_b, BLOCK * l_b + l_b) 
        B_O = range(BLOCK * l_b - l_b,BLOCK * l_b)
        
        start = timer()
        # Compensation
        for IN in range(NMux):
            for OUT in range(NMux):        
                X_k = np.zeros(l_b*2,dtype = np.complex128)
                for S in range(ovsmpl):
                    B_IOS = range(BLOCK * l_b * ovsmpl - l_b * ovsmpl + S , BLOCK * l_b * ovsmpl + l_b * ovsmpl + S,ovsmpl) 
                    X_i[IN,S,:] = np.fft.fft(x[IN,B_IOS])
                   
                X_k = X_k + X_i[IN,S,:] * H[IN,OUT,S,:]           

                 
            x_o[OUT,B_O] = x_o[OUT,B_O] + (np.fft.ifft(X_k)[SecondBlock])
        times[0] = min([times[0],timer()-start])
        # Tap updates
        t_ECalc = 0  
        t_GC = 0
        for IN in range(NMux):
            for OUT in range(NMux):        
                for S in range(ovsmpl):
                    start = timer()                    
                    e = (I - x_o[OUT,B_O] * np.conj(x_o[OUT,B_O])) * x_o[OUT,B_O]
                    E = np.fft.fft(np.append(O,e)) 
                    t_ECalc += timer()-start
                    
                    start = timer()                    
                    s_ = np.fft.ifft(np.conj(X_i[IN,S,:]) * E)[FirstBlock]
                    S_ = np.fft.fft(np.append(s_,O))
                    t_GC += timer()-start
                    
                    start = timer()                    
                    H[IN,OUT,S,:] = H[IN,OUT,S,:] + mu * S_
                    t_ECalc += timer()-start
                    
        times[1] = min([times[1],t_ECalc])
        times[2] = min([times[2],t_GC])
    
    return times



import numpy as np
import pyculib.fft as GFFT
import numba.cuda as cuda
from MIMOKernels import *
from timeit import default_timer as timer


def FDE_MIMO_GPU(x,l_b,mu,ovsmpl,R2 = 2):

    # Preparations
    C_Type = x.dtype
    NSyms =int(len(x[0,:])/ovsmpl)
    NMux = len(x[:,0])

    x_o = np.zeros(NMux*NSyms,dtype = C_Type).reshape(NMux,NSyms)
    NBlocks = int(NSyms/l_b - 1)
    
    # Setup Filter Variables
    
    O = np.zeros(l_b,dtype = C_Type)
    I = np.ones(l_b,dtype = C_Type) * R2 
    
    h = np.zeros(l_b * 2 * NMux * NMux,dtype = C_Type).reshape(NMux,NMux,l_b * 2)
    H = np.zeros(l_b * 2 * NMux * NMux * ovsmpl,dtype = C_Type).reshape(NMux,NMux,ovsmpl,l_b * 2)

    CTap = np.int(l_b / 2)
    for OUT in range(NMux):
        h[OUT,OUT,CTap] = 1/ovsmpl + 0j
    for IN in range(NMux):
        for OUT in range(NMux):
            for S in range(ovsmpl):
                H[IN,OUT,S,:] = np.fft.fft(h[IN,OUT,:])

    H = H.reshape(NMux*NMux*ovsmpl*l_b*2)

    # Intermediate values
    X_i = np.zeros(NMux*ovsmpl*l_b*2,dtype = C_Type)
    X_o = np.zeros(NMux*NMux*l_b*2,dtype = C_Type)

    E = np.zeros(l_b * 2 * NMux * NMux * ovsmpl,dtype = C_Type)
    s = np.zeros(l_b * 2 * NMux * NMux * ovsmpl,dtype = C_Type)


    # FFT planning
    DBlock_IN_FFT = GFFT.FFTPlan((l_b*2,),C_Type,C_Type,NMux*ovsmpl)
    DBlock_IN_OUT_FFT = GFFT.FFTPlan((l_b*2,),C_Type,C_Type,NMux*NMux)
    DBlock_IN_OUT_S_FFT = GFFT.FFTPlan((l_b*2,),C_Type,C_Type,NMux*NMux*ovsmpl)

    ## Transmit TO GPU
    
    I = cuda.to_device(I)
    s= cuda.to_device(s)
    H = cuda.to_device(H)
    X_o = cuda.to_device(X_o)
    E = cuda.to_device(E)
    x_o = cuda.to_device(x_o)
    X_i = cuda.to_device(X_i)
    x = cuda.to_device(x)
    
    # Start
    print(NBlocks)
    for BLOCK in range(1,NBlocks):
        
        # Compensation
        GetBlocks[(NMux,ovsmpl),l_b*2](x,X_i,BLOCK,l_b,ovsmpl)
        DBlock_IN_FFT.forward(X_i,X_i)
        Compensate[(NMux,NMux),l_b*2](X_i,X_o,H,l_b,ovsmpl,NMux)

        DBlock_IN_OUT_FFT.inverse(X_o,X_o)
        Reduce[(NMux,NMux),l_b*2](X_o,x_o,NMux,l_b,BLOCK)
        # Tap Updates
        CalcE[(NMux,NMux,ovsmpl),l_b](x_o,E,NMux,l_b,BLOCK,I,ovsmpl)
        DBlock_IN_OUT_S_FFT.forward(E,E) 
        CompensateWithInput[(NMux,NMux,ovsmpl),l_b*2](E,X_i,l_b,NMux,ovsmpl)
        
  
        DBlock_IN_OUT_S_FFT.inverse(E,s)
        OverlapS[(NMux,NMux,ovsmpl),l_b](s,l_b,NMux,ovsmpl)
        #print("s")
        #print(s.copy_to_host())
  
        DBlock_IN_OUT_S_FFT.forward(s,s)
        UpdateH[(NMux,NMux,ovsmpl),l_b*2](H,s,mu,l_b,NMux,ovsmpl)

       
    return x_o.copy_to_host()

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


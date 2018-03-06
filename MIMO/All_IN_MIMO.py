import numpy as np
import pyculib.fft as GFFT
import numba.cuda as cuda
from MIMOKernels import *
from timeit import default_timer as timer

def ALL_IN_MIMO(x,l_b,mu,ovsmpl,R2 = 2):
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
    DBlock_IN_S_FFT = GFFT.FFTPlan((l_b*2,),C_Type,C_Type,NMux*ovsmpl*NBlocks)
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
    ovsmpl_OvlapSamps = (NBlocks) * l_b*2
    mux_OvlapSamps = ovsmpl_OvlapSamps * ovsmpl
    x_i = np.zeros(NMux*mux_OvlapSamps, dtype = C_Type)
    

    GetAllBlocks[(NBlocks,NMux,ovsmpl),l_b*2](x,x_i,l_b,NSyms,ovsmpl,ovsmpl_OvlapSamps,mux_OvlapSamps)
    DBlock_IN_S_FFT.forward(x_i,x_i)
    
    
    for BLOCK in range(1,NBlocks):
        # Compensation
                        
        Compensate_ALLIN[(NMux,NMux),l_b*2](x_i,X_o,H,l_b,ovsmpl,ovsmpl_OvlapSamps,mux_OvlapSamps,BLOCK,NMux)
        DBlock_IN_OUT_FFT.inverse(X_o,X_o)
        CalcE_SUM[(NMux,NMux,ovsmpl),l_b*2](x_o,X_o,E,NMux,l_b,BLOCK,I,ovsmpl)
        
        DBlock_IN_OUT_S_FFT.forward(E,E) 
        CompensateWithInput_ALL_IN[(NMux,NMux,ovsmpl),l_b*2](E,x_i,BLOCK,l_b,NMux,ovsmpl,ovsmpl_OvlapSamps,mux_OvlapSamps)
    
        DBlock_IN_OUT_S_FFT.inverse(E,s)
        OverlapS[(NMux,NMux,ovsmpl),l_b](s,l_b,NMux,ovsmpl)
        DBlock_IN_OUT_S_FFT.forward(s,s)
        UpdateH[(NMux,NMux,ovsmpl),l_b*2](H,s,mu,l_b,NMux,ovsmpl)

       
    return x_o.copy_to_host()


def ALL_IN_MIMO_SpeedTest(x,l_b,mu,ovsmpl,R2 = 2):
# Preparations
    C_Type = x.dtype
    NSyms =int(len(x[0,:])/ovsmpl)
    NMux = len(x[:,0])
    stream = cuda.stream()

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
    DBlock_IN_S_FFT = GFFT.FFTPlan((l_b*2,),C_Type,C_Type,NMux*ovsmpl*NBlocks,stream = stream)
    DBlock_IN_OUT_FFT = GFFT.FFTPlan((l_b*2,),C_Type,C_Type,NMux*NMux,stream = stream)
    DBlock_IN_OUT_S_FFT = GFFT.FFTPlan((l_b*2,),C_Type,C_Type,NMux*NMux*ovsmpl,stream = stream)
     
    ## Transmit TO GPU
    
    I = cuda.to_device(I,stream = stream)
    s= cuda.to_device(s,stream = stream)
    H = cuda.to_device(H,stream = stream)
    X_o = cuda.to_device(X_o,stream = stream)
    E = cuda.to_device(E,stream = stream)
    x_o = cuda.to_device(x_o,stream = stream)
    X_i = cuda.to_device(X_i,stream = stream)
    x = cuda.to_device(x,stream = stream)
    
    # Start
    ovsmpl_OvlapSamps = (NBlocks) * l_b*2
    mux_OvlapSamps = ovsmpl_OvlapSamps * ovsmpl
    x_i = np.zeros(NMux*mux_OvlapSamps, dtype = C_Type)
    x_i = cuda.to_device(x_i)

    GetAllBlocks[(NBlocks,NMux,ovsmpl),l_b*2](x,x_i,l_b,NSyms,ovsmpl,ovsmpl_OvlapSamps,mux_OvlapSamps)
    DBlock_IN_S_FFT.forward(x_i,x_i)
    
    t_comp = 90
    t_fft_inv_1 = 90
    t_Reduce = 90
    t_Calc_E = 90
    t_FFT_EE = 90
    t_compwXI = 90
    t_IFFT_ES = 90
    t_OVLP = 90
    t_FFT_SS = 90
    t_UpdateH = 90
    for BLOCK in range(1,NBlocks):
        # Compensation
        start = timer()         
        Compensate_ALLIN[(NMux,NMux),l_b*2](x_i,X_o,H,l_b,ovsmpl,ovsmpl_OvlapSamps,mux_OvlapSamps,BLOCK,NMux)
        t_comp = min([t_comp, timer()-start])
        
        start = timer()         
        DBlock_IN_OUT_FFT.inverse(X_o,X_o)
        t_fft_inv_1 = min([t_fft_inv_1 , timer()-start])

        start = timer()         
        Reduce[(NMux,NMux),l_b*2](X_o,x_o,NMux,l_b,BLOCK)
        t_Reduce = min([t_Reduce , timer()-start])
        
        
        # Tap Updates
        start = timer()         
        CalcE_SUM[(NMux,NMux,ovsmpl),l_b*2](x_o,X_o,E,NMux,l_b,BLOCK,I,ovsmpl)
        t_Calc_E = min([t_Calc_E , timer()-start])
        
        start = timer()         
        DBlock_IN_OUT_S_FFT.forward(E,E) 
        t_FFT_EE = min([t_FFT_EE , timer()-start])

        start = timer()         
        CompensateWithInput_ALL_IN[(NMux,NMux,ovsmpl),l_b*2](E,x_i,BLOCK,l_b,NMux,ovsmpl,ovsmpl_OvlapSamps,mux_OvlapSamps)
        t_compwXI = min([t_compwXI , timer()-start])

        start = timer()         
        DBlock_IN_OUT_S_FFT.inverse(E,s)
        t_IFFT_ES = min([t_IFFT_ES , timer()-start])
        
        start = timer()         
        OverlapS[(NMux,NMux,ovsmpl),l_b](s,l_b,NMux,ovsmpl)
        t_OVLP = min([t_OVLP , timer()-start])
        
        start = timer()         
        DBlock_IN_OUT_S_FFT.forward(s,s)
        t_FFT_SS = min([t_FFT_SS , timer()-start])
        
        start = timer()         
        UpdateH[(NMux,NMux,ovsmpl),l_b*2](H,s,mu,l_b,NMux,ovsmpl)
        t_UpdateH = min([t_UpdateH , timer()-start])
       
    return np.array([t_comp,t_fft_inv_1,t_Calc_E,t_FFT_EE,t_compwXI,t_IFFT_ES,t_OVLP,t_FFT_SS,t_UpdateH])

import numpy as np
import numba.cuda as cuda
def deMuxSamples(x,ovsmpl,C_Type):
    NSamps = len(x[0,:])
    NMux = len(x[:,0])
    y = np.empty(NMux*NSamps,dtype = C_Type).reshape(NMux,ovsmpl,int(NSamps/ovsmpl))
    
    for c in range(NMux):
        for s in range(ovsmpl):
            sampleRange = range(s,NSamps, ovsmpl)
            y[c,s,:] = x[c,sampleRange]         
    return y

@cuda.jit
def GetBlocks(x,X_i,BLOCK,l_b,ovsmpl):
    
    tid = cuda.threadIdx.x
    IN = cuda.blockIdx.x
    S = cuda.blockIdx.y

    SampleId = tid * ovsmpl + S + BLOCK * l_b * ovsmpl 
    IS_ID = tid + l_b* 2 * S  + l_b * 2 * ovsmpl * IN 
    X_i[IS_ID] = x[IN,SampleId]

@cuda.jit
def GetAllBlocks(x,X_i,l_b,NSyms,ovsmpl,ovsmpl_OvlapSamps,mux_OvlapSamps):
    
    tid = cuda.threadIdx.x # which symbol
    BLOCK = cuda.blockIdx.x # Which Block
    IN = cuda.blockIdx.y # Which channel
    s = cuda.blockIdx.z # Which sample (Even/odd) symbol
    
    SampleId = tid * ovsmpl + s + BLOCK * l_b * ovsmpl
    FullId = tid + BLOCK * l_b * 2 + s * ovsmpl_OvlapSamps + IN * mux_OvlapSamps 

    X_i[FullId] = x[IN,SampleId]

@cuda.jit
def Compensate_ALLIN(X_i,X_o,H,l_b,ovsmpl,ovsmpl_OvlapSamps,mux_OvlapSamps,BLOCK,NMux):

    tid = cuda.threadIdx.x
    IN = cuda.blockIdx.y
    OUT = cuda.blockIdx.x

    
    val = 0
    for s in range(ovsmpl):
        I_Id = tid + BLOCK * l_b * 2 + s * ovsmpl_OvlapSamps + IN * mux_OvlapSamps 
        IOS_ID = tid + l_b* 2 * s  + l_b * 2 * ovsmpl * IN +  l_b* 2 * ovsmpl * NMux * OUT  
        IO_Id = tid + l_b* 2 * IN +  l_b * 2 * NMux * OUT 
        val+= X_i[I_Id]* H[IOS_ID]
        
    X_o[IO_Id] = val

@cuda.jit
def Compensate(X_i,X_o,H,l_b,ovsmpl,NMux):

    tid = cuda.threadIdx.x
    IN = cuda.blockIdx.y
    OUT = cuda.blockIdx.x

    
    val = 0
    for s in range(ovsmpl):

        IS_ID = tid + l_b* 2 * s  + l_b * 2 * ovsmpl * IN 
        IO_ID = tid +  l_b * 2 * IN +  l_b* 2 * NMux * OUT  
        
        IOS_ID = tid + l_b* 2 * s  + l_b * 2 * ovsmpl * IN +  l_b* 2 * ovsmpl * NMux * OUT  
        val = val + X_i[IS_ID]* H[IOS_ID]
        
    X_o[IO_ID] = val


@cuda.jit
def Reduce(X_o,x_o,NMux,l_b,BLOCK):
    tid = cuda.threadIdx.x
    OUT = cuda.blockIdx.x
    val = 0
    
    for IN in range(NMux):
        IO_ID = tid + l_b + l_b * 2 * IN + OUT * NMux * l_b * 2 
        # + l_b to skip first block
        val += X_o[IO_ID]

    OUTID = BLOCK*l_b - l_b + tid
    x_o[OUT,OUTID] = val/(l_b*2)
    #/(l_b*2 to normalize IFFT)


# Optimized a bit
@cuda.jit
def CalcE_SUM(x_o,X_o,E,NMux,l_b,BLOCK,I,ovsmpl):
    tid = cuda.threadIdx.x
    IN = cuda.blockIdx.x    
    OUT = cuda.blockIdx.y
    S = cuda.blockIdx.z # Which sample (Even/odd) symbol
    OUTID = BLOCK*l_b - l_b + tid
    val = 0
    for IN2 in range(NMux):
        IO_ID = tid + l_b + l_b * 2 * IN2 + OUT * NMux * l_b * 2 
        # + l_b to skip first block
        val += X_o[IO_ID]
    out = val/(2*l_b)

    IOS_ID = tid + l_b* 2 * S  + l_b * 2 * ovsmpl * IN +  l_b* 2 * ovsmpl * NMux * OUT 

        # Appending zeros
    # Forgive me for this if statement here
    if tid<l_b:

        IOS_ID = tid + l_b* 2 * S  + l_b * 2 * ovsmpl * IN +  l_b* 2 * ovsmpl * NMux * OUT 
        conj = out.real + out.imag * -1j
        e = (I[tid] - out * conj) * out
        
        E[IOS_ID + l_b] = e
    else:
        E[IOS_ID - l_b] = 0 + 0j
         

    x_o[OUT,OUTID] = out
    
    

@cuda.jit
def CalcE(x_o,E,NMux,l_b,BLOCK,I,ovsmpl):
    tid = cuda.threadIdx.x
    
    IN = cuda.blockIdx.x
    OUT = cuda.blockIdx.y
    S = cuda.blockIdx.z # Which sample (Even/odd) symbol


    OUTID = BLOCK * l_b - l_b + tid
    IOS_ID = tid + l_b* 2 * S  + l_b * 2 * ovsmpl * IN +  l_b* 2 * ovsmpl * NMux * OUT 
    conj = x_o[OUT,OUTID].real + x_o[OUT,OUTID].imag * -1j
    e = (I[tid] - x_o[OUT,OUTID] * conj) * x_o[OUT,OUTID]
    
    # Appending zeros
    E[IOS_ID] = 0 + 0j
    E[IOS_ID + l_b] = e


@cuda.jit
def CompensateWithInput(E,X_i,l_b,NMux,ovsmpl):
    tid = cuda.threadIdx.x
    IN = cuda.blockIdx.x
    OUT = cuda.blockIdx.y
    S = cuda.blockIdx.z


    IOS_ID = tid + l_b* 2 * S  + l_b * 2 * ovsmpl * IN +  l_b* 2 * ovsmpl * NMux * OUT 
    IS_ID = tid + l_b* 2 * S  + l_b * 2 * ovsmpl * IN 

    E[IOS_ID] = E[IOS_ID] * (X_i[IS_ID].real + X_i[IS_ID].imag * -1j)


@cuda.jit
def CompensateWithInput_ALL_IN(E,X_i,BLOCK,l_b,NMux,ovsmpl,ovsmpl_OvlapSamps,mux_OvlapSamps):
    tid = cuda.threadIdx.x
    IN = cuda.blockIdx.x
    OUT = cuda.blockIdx.y
    S = cuda.blockIdx.z # Which sample (Even/odd) symbolC

    IOS_ID = tid + l_b* 2 * S  + l_b * 2 * ovsmpl * IN +  l_b* 2 * ovsmpl * NMux * OUT  
    FullId = tid + BLOCK * l_b * 2 + S * ovsmpl_OvlapSamps + IN * mux_OvlapSamps 
    
    IO_ID = l_b*2 * IN + OUT * NMux * l_b*2  + tid
    E[IOS_ID] = E[IOS_ID] * (X_i[FullId].real + X_i[FullId].imag * -1j)


@cuda.jit
def OverlapS(s,l_b,NMux,ovsmpl):
    tid = cuda.threadIdx.x
    IN = cuda.blockIdx.x
    OUT = cuda.blockIdx.y
    S = cuda.blockIdx.z # Which sample (Even/odd) symbol

    IOS_ID = tid + l_b* 2 * S  + l_b * 2 * ovsmpl * IN +  l_b* 2 * ovsmpl * NMux * OUT  
    s[IOS_ID] = s[IOS_ID]/(l_b*2)
    s[IOS_ID + l_b] = 0 + 0j 


@cuda.jit
def UpdateH(H,s,mu,l_b,NMux,ovsmpl):
    tid = cuda.threadIdx.x
    IN = cuda.blockIdx.x
    OUT = cuda.blockIdx.y
    S = cuda.blockIdx.z # Which sample (Even/odd) symbol

    IOS_ID = tid + l_b* 2 * S  + l_b * 2 * ovsmpl * IN +  l_b* 2 * ovsmpl * NMux * OUT
    H[IOS_ID] = H[IOS_ID] + mu * s[IOS_ID]


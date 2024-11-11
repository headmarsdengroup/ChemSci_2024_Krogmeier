import scipy.linalg as scp
import numpy as np
from functools import reduce
import dependencies.constants as const

def fully_coh_state(size):
    '''
    Generates a fully entangled pure state.
    '''
    num = 1/size
    fcs = np.ones([size,size],dtype=np.complex128)*num
    return fcs

def tensor_chain(spin_types,index,op):
    
    input_chain = []
    for i,spin in enumerate(spin_types):
        
        if i == index: 
            input_chain.append(const.spin_operators[const.spin_type_dict[spin]][op])
        elif i != index:
            input_chain.append(const.spin_operators[const.spin_type_dict[spin]]['Id'])
    
    return reduce(np.kron, input_chain)

def rot_op(spin_types):
    
    '''
    Builds the Rotation Operator for a spin echo experiment.
    '''
    
    return scp.expm(-1j*(np.pi)/(1)*(tensor_chain(spin_types,0,'sx')))

def init_dens(spin_types):
    
    '''
    Generates the initial density matrix for a fully entangled initial state.
    Takes as input:
    - the list of spin types in the system.
    '''
    
    input_chain = []
    for spin in spin_types:
        input_chain.append(fully_coh_state(int(2*const.spin_type_dict[spin]+1)))
    
    return reduce(np.kron, input_chain)

def svec(index,spin_types):
    svec = np.array([tensor_chain(spin_types,index,'sx'),tensor_chain(spin_types,index,'sy'),tensor_chain(spin_types,index,'sz')])
    return svec

def compute_SAI(svec,ivec,A,dim,spin_types):
    
    '''
        Takes two 3,2^N,2^N vectors, and computes the SAI product with a 3x3 hyperfine tensor A
    '''
    
    SA = np.zeros([3,dim,dim],dtype=np.complex128)
    for index_1,col in enumerate(range(A.shape[1])):
        temp = np.zeros([dim,dim],dtype=np.complex128) 
        for index_2,i in enumerate(A[:,col]):
            temp += svec[index_2] * i
        SA[index_1] = temp

    SAI = np.zeros([dim,dim],dtype=np.complex128)
    for i,j in zip(SA,ivec):
        SAI += i@j
        
    return SAI

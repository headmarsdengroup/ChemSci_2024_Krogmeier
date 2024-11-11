import numpy as np
import scipy.linalg as scp

def vec_lb(rates,Ls,H):
    '''
    takes flattened density matrix, list of rates, list of channels, the Hamiltonian, as input
    returns a superoperator that can then be used to propagate the density matrix through 
    matrix exponentiation
    '''
    dim = len(H)
    Iden = np.eye(dim,dtype=np.complex128)
    
    L = -1j*np.kron(Iden,H) + 1j*(np.kron(H.T,Iden))
    
    for op,rate in zip(Ls,rates):
        L += rate*np.kron(np.conjugate(op),op) - \
        (rate/2)*(np.kron(Iden,(np.conjugate(op.T)@op))) - \
        (rate/2)*(np.kron(((op.T)@np.conjugate(op)),Iden))
        
    return L

def prop_vec_LB(times,dim,rho,rates,Ls,H1,H2,showtime=False,Hahn=False):

    '''
    propagates the density matrix according to the vectorized Lindblad equation
    '''

    if Hahn:
        Hahn = np.zeros(len(times))
        for t in range(len(times)):
            if showtime:
                print(f'{t + 1} out of {len(times)}')
            p0 = rho.flatten()
            pt1 = scp.expm((times[t]/2)*vec_lb(rates,Ls,H1))@p0
            pt2 = scp.expm((times[t]/2)*vec_lb(rates,Ls,H2))@pt1
            rhot = pt2.reshape((int(np.sqrt(len(pt2))),int(np.sqrt(len(pt2)))))
            redrho = np.trace(rhot.reshape(2,int(dim/2),2,int(dim/2)),axis1=1,axis2=3)
            Hahn[t] = redrho[0][1]/0.5
        return Hahn
    elif not Hahn:
        FID = np.zeros(len(times))
        for t in range(len(times)):
            if showtime:
                print(f'{t + 1} out of {len(times)}')
            p0 = rho.flatten()
            pt1 = scp.expm(times[t]*vec_lb(rates,Ls,H1))@p0
            rhot = pt1.reshape((int(np.sqrt(len(pt1))),int(np.sqrt(len(pt1)))))
            redrho = np.trace(rhot.reshape(2,int(dim/2),2,int(dim/2)),axis1=1,axis2=3)
            FID[t] = redrho[0][1]/0.5
        return FID
import numpy as np
from itertools import combinations
import dependencies.misc as misc
import dependencies.constants as const
import dependencies.hamiltonian as ham
import dependencies.mefuncs as lbs

def average_others(cluster_key,imap,spin_dict):
    '''
    Takes as input:
    - the key of the cluster being considered
    - the total imap
    - the total spin dictionary
    
    Returns:
    - the average location of the bath spins
    - the average dipolar couplings between baths in the cluster and all bath spins not in the cluster
    - the average hyperfine interaction of the electron with all bath spins not in the cluster
    '''
    spin_keys = list(spin_dict.keys())
    other_keys = []
    for key in spin_keys:
        if key not in cluster_key and key != 0:
            other_keys.append(key)
    
    num_others = len(other_keys)
    
    avg_loc = np.array([0,0,0],dtype=np.float64)
    for key in other_keys:
        avg_loc += spin_dict[key][0]
    avg_loc /= num_others
    
    avg_jijs = {}
    for spin in cluster_key:
        avg_imap = np.zeros([3,3],dtype=np.complex128)
        for other in other_keys:
            if spin < other:
                if imap[f'({spin}, {other})'].shape == (3, 3):
                    avg_imap += imap[f'({spin}, {other})']
                    bnm = False
                elif imap[f'({spin}, {other})'].shape != (3, 3):
                    avg_imap[0][0] += imap[f'({spin}, {other})']
                    bnm = True
            elif spin > other:
                if imap[f'({other}, {spin})'].shape == (3, 3):
                    avg_imap += imap[f'({other}, {spin})']
                    bnm = False
                elif imap[f'({other}, {spin})'].shape != (3, 3):
                    avg_imap[0][0] += imap[f'({other}, {spin})']
                    bnm = True 
        if not bnm:
            avg_imap /= num_others 
            avg_jijs[spin] = avg_imap
        elif bnm:
            avg_imap[0][0] /= num_others 
            avg_jijs[spin] = avg_imap[0][0]
    
    
    avg_hyp = np.zeros([3,3],dtype=np.complex128)
    for other in other_keys:
        if imap[f'(0, {other})'].shape == (3, 3):
            avg_hyp += imap[f'(0, {other})']
            bnm = False
        elif imap[f'(0, {other})'].shape != (3, 3):
            avg_hyp[0][0] += imap[f'(0, {other})']
            bnm = True
    if not bnm:
        avg_hyp /= num_others
    elif bnm:
        avg_hyp[0][0] /= num_others
        avg_hyp = avg_hyp[0][0]
        
    return avg_loc, avg_jijs, avg_hyp

def factorize(imap,order,numSpins,spin_types,spin_dict):
    
    '''
    Factorizes a system into clusters, while averaging the interactions with the rest of the bath. 
    Takes as input:
    - the interaction mapping of the system.
    - the size of the clusters (order)
    - the total number of spins in the system
    - a list of the spin types in the system
    - the spin dictionary object
    '''
    
    clusters = {}
    cluster_types = {}
    if order > len(range(1,numSpins)):
        print(f'no clusters of size {order}')
        order = len(range(1,numSpins)) - 1
    elif order == len(range(1,numSpins)):
        print(f'cluster of size {order} does not require factorization. This is exact diagonalization')
        order = len(range(1,numSpins)) - 1
    intes = list(range(1,numSpins))
    interactions = combinations(intes,order)
    for inte_ in interactions:
        clusters[inte_] = {}
        cluster_types[inte_] = {}
    keys = list(clusters.keys())
    for key in keys:
        al, aj, ah = average_others(key,imap,spin_dict)
        cluster_types[key]['types'] = ['e']
        clusters[key]['imap'] = {}
        clusters[key]['spin_dict'] = {}
        clusters[key]['spin_dict'][0] = spin_dict[0]
        dip_ints = combinations(key,2)
        for int_ in dip_ints:
            clusters[key]['imap'][f'{int_}'] = imap[f'{int_}']
        for spin in key:
            cluster_types[key]['types'].append(spin_types[spin])
            clusters[key]['imap'][f'(0, {spin})'] = imap[f'(0, {spin})']
            clusters[key]['spin_dict'][spin] = spin_dict[spin]
        cluster_types[key]['types'].append('H') # adding the average bath spin not included in the cluster
        clusters[key]['spin_dict'][int(numSpins + 1)] = [al,const.gyro_dict['H']]
        for spin in aj: # adding the interactions of the cluster nuclei with the averaged spin
            clusters[key]['imap'][f'({spin}, {int(numSpins + 1)})'] = aj[spin]
        clusters[key]['imap'][f'(0, {int(numSpins + 1)})'] = ah # adds the averaged hyperfine interaction with the rest of the nuclei not in the cluster.
        
        dim = 1
        for type_ in cluster_types[key]['types']:
            dim *= 2*const.spin_type_dict[type_] + 1
        cluster_types[key]['dim'] = int(dim)
        
    return clusters, cluster_types

def cce_2nd_order(imap,numSpins,spin_types,spin_dict,Bext,times,gammas):

    '''
    Performs a 2nd order cce factorization of the bath. 
    Takes as input:
    - the total interaction map
    - the total number of spins in the system
    - the list of all spin types in the system
    - the total spin dictionary
    - the external field
    - the time range of the calculation
    
    Note: Does not work for clusters smaller or larger than two.
    '''
    
    order = 2

    cluster_coh_masterlist = {}
    for order_ in range(1, order + 1):
        clusters, cluster_info = factorize(imap,order_,numSpins,spin_types,spin_dict)
        for cluster,info in zip(clusters,cluster_info):
            rho = misc.init_dens(cluster_info[info]['types'])
            Rot = misc.rot_op(cluster_info[info]['types'])
            H_hyp = ham.H_hf(clusters[cluster]['imap'],cluster_info[info]['dim'],cluster_info[info]['types'])
            H_B = ham.H_mag(cluster_info[info]['dim'],len(cluster_info[info]['types']),clusters[cluster]['spin_dict'],Bext[2],cluster_info[info]['types'])
            H_tot = ham.H_total(H_hyp,H_B)

           
                
            rates = []
            Ls = []
                
            if order_ == 1:
                rates.append(0)
                Ls.append(misc.tensor_chain(cluster_info[info]['types'],0,'sz'))
                
            elif order_ == 2:
                rates.append(gammas[f'{cluster}'])
                Ls.append(misc.tensor_chain(cluster_info[info]['types'],0,'sz'))
                    
            H1 = H_tot
            H2 = Rot@H1@np.conjugate(Rot.T)
                    
            prop = lbs.prop_vec_LB(times,cluster_info[info]['dim'],rho,rates,Ls,H1,H2,showtime=False,Hahn=True)
                
            cluster_coh_masterlist[cluster] = prop
                
    coh = np.ones(len(times),dtype=np.complex128)
    for clust in cluster_coh_masterlist:
        if len(clust) == order:
            prod_prev = np.ones(len(times),dtype=np.complex128)
            for sub_clust in clust:
                prod_prev *= cluster_coh_masterlist[(sub_clust,)]
            coh *= cluster_coh_masterlist[clust]/prod_prev
            
    return coh

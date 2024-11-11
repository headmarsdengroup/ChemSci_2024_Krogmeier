import numpy as np
import os
import re
from itertools import combinations
from scipy.stats import norm
import dependencies.spin_dict as sd

def seperate_string_number(string):
    previous_character = string[0]
    groups = []
    newword = string[0]
    for x, i in enumerate(string[1:]):
        if i.isalpha() and previous_character.isalpha():
            newword += i
        elif i.isnumeric() and previous_character.isnumeric():
            newword += i
        else:
            groups.append(newword)
            newword = i

        previous_character = i

        if x == len(string) - 2:
            groups.append(newword)
            newword = ''
    return groups

def HFfromORCA(filename):

    '''
    Takes an ORCA output file as input, and returns a dictionary
    with the nuclei and their associated hyperfine tensors.
    If it's not working, check the ORCA printlevel; set it to 2. 
    Note: ORCA tensors are in MHz, so they need to be multiplied by 1000 
    to match up with the units used in the rest of the code.
    Added in multiplication by 1000.
    '''

    with open(filename,'r') as f:
        for line in f:
            if 'ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE' in line:
                break
        with open('newfile.txt','a') as outfile:
            outfile.writelines(f.readlines())

    keys = []
    values = []
    with open('newfile.txt','r') as f:
        for line in f:
            if 'Nucleus' in line:
                key = line.split()[1]
                keys.append(key)
                
            if 'Raw HFC matrix' in line:
                stuff = []
                for i in range(4):
                    stuff.append(next(f).split())
                hf = []
                for j in stuff:
                    if len(j) == 3:
                        hf.append(j)
                hf = np.array(hf).astype(float)
                values.append(hf*1000)
                
    os.remove('newfile.txt')
    values = values[-1:] + values[0:-1] # necessary so the order of the xyzs and hfs is the same
    keys = keys[-1:] + keys[0:-1] # necessary so the order of the xyzs and hfs is the same

    dict_1 = {key:value for key,value in zip(keys,values)}
    
    nuclei = list(dict_1.keys())
    
    new_keys = []
    new_values = []
    for i, key in enumerate(list(dict_1.keys())):
        new_keys.append(i + 1)
        new_values.append(dict_1[key])
       
    tmp_dict = {key:value for key, value in zip(new_keys,new_values)}
        
    hfs_dict = {}
    for index,key in enumerate(list(tmp_dict.keys())):
        hfs_dict[index + 1] = tmp_dict[key]   
    
    return nuclei, hfs_dict
    
def EFGfromORCA(filename):

    '''
    Returns the electric field gradients.
    '''

    with open(filename,'r') as f:
        for line in f:
            if 'ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE' in line:
                break
        with open('newfile.txt','a') as outfile:
            outfile.writelines(f.readlines())

    a = {}
    keys = []
    with open('newfile.txt','r') as f:
        for line in f:
            if 'Nucleus' in line:
                keys.append(line.split()[1])

    efgs = []
    with open('newfile.txt','r') as f:
        for line in f:
            if 'Raw EFG matrix' in line:
                stuff = []
                for i in range(6):
                    stuff.append(next(f).split())
                efg = []
                for j in stuff:
                    if len(j) == 3:
                        efg.append(j)
                efgs.append(np.array(efg).astype(float))

    key_value_pairs = list(zip(keys,efgs))

    os.remove('newfile.txt')

    return {k: v for k, v in key_value_pairs}

def getCoords(xyzfile,nuclei):
    
    '''
    Returns a dictionary with keys as nuclei and xyz coordinates as values.
    Takes as input:
    - the name of the xyz file
    - the list of spin-active nuclei
    
    Notes: As written, requires hyperfine tensors to be already generated. This could be altered if necessary.
    '''

    coords = np.genfromtxt(xyzfile, skip_header=2, dtype='unicode')
    
    spin_type = []
    for i in nuclei:
        if seperate_string_number(i)[1] not in spin_type:
            spin_type.append(seperate_string_number(i)[1])
    
    keys = []
    for ind,i in enumerate(coords[:,0]):
        if i in spin_type:
            keys.append(str(ind) + i)
    
    xyzs = []
    for coord in coords:
        if coord[0] in spin_type:
            xyzs.append(np.array(coord[1:4]).astype(float))

    key_value_pairs = list(zip(keys,xyzs))
    
    return {k: v for k, v in key_value_pairs}

def avgeProcess(soi,dir_name):
    
    fnumber = len([name for name in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, name))])
    file_numbers = range(0,fnumber)

    spins_of_interest = soi
    xyzs = {}
    xyzs[dir_name] = {}
    for file in file_numbers:
        xyzs[dir_name][f'geom_{file}'] = {}
        try:
            with open(f'{dir_name}/output_{file}.log','r') as log:
                for line in log:
                    if '* xyz -2 2' in line:
                        for index,i in enumerate(range(60)):
                            sline = next(log).split()
                            if len(sline) == 6:
                                if sline[2] in spins_of_interest:
                                    xyzs[dir_name][f'geom_{file}'][f'{index}{sline[2]}'] = np.array([float(sline[3]),float(sline[4]),float(sline[5])])
        except:
            pass
            
    hfs = {}
    hfs[dir_name] = {}
    for file in file_numbers:
        try:
            with open(f'{dir_name}/output_{file}.log','r') as log:
                for line in log:
                    if 'ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE' in line:
                        break
                with open('newfile.txt','a') as outfile:
                    outfile.writelines(log.readlines())
            keys = []
            values = []
            with open('newfile.txt','r') as f:
                for line in f:
                    if 'Nucleus' in line:
                        key = line.split()[1]
                        keys.append(key)
                
                    if 'Raw HFC matrix' in line:
                        stuff = []
                        for i in range(4):
                            stuff.append(next(f).split())
                        hf = []
                        for j in stuff:
                            if len(j) == 3:
                                hf.append(j)
                        hf = np.array(hf).astype(float)
                        values.append(hf*1000)
    
            os.remove('newfile.txt')
            if values == []:
                values = np.zeros([13,3,3]) # why 13 here?
                keys = [x for x in range(0,13)]
            values = values[-1:] + values[0:-1] # necessary so the order of the xyzs and hfs is the same
            keys = keys[-1:] + keys[0:-1] # necessary so the order of the xyzs and hfs is the same
            hfs[dir_name][f'geom_{file}'] = {key:value for key,value in zip(keys,values)}
        except:
            pass
    
    bad_hfs = []
    for coord, hf in zip(xyzs[dir_name],hfs[dir_name]):
        key_check = list(hfs[dir_name][hf].keys())
        if type(key_check[0]) == int:
            bad_hfs.append([dir_name,hf])
        
    for bad in bad_hfs:
        xyzs[bad[0]].pop(bad[1])
        hfs[bad[0]].pop(bad[1])
        
    return xyzs, hfs

#
#
### Everything under here still needs work
#
#

def ff_rates_from_Sousa2003(imap,spin_dict,sdb=True):
    
    '''
    Calculates the flip-flop rate of nuclei in the system, following the derivation in de Sousa, 2003: PhysRevB.68.115322.
    Takes as input:
    - the interaction map calculated using Hamiltonian_main.py
    - the spin dictionary object
    Optional parameter:
    sdb=True will calculate the hyperfine energy difference of the two nuclei undergoing the flip-flop process.
    '''

    zhat = np.array([0,0,1])
    I = 1/2
    AI = (2/15)*((I*(I + 1))/(2*I + 1))*(2*I*(I + 1) + 1)
    prefactor = 2*np.sqrt(2*np.pi)*AI

    intmap = []
    for inte in imap:
        nums = re.findall(r'\d+', inte)
        if int(nums[0]) != 0:
            intmap.append([int(nums[0]),int(nums[1])])
                 
    Zdif_squared = {}

        
    for inte in intmap:
        if imap[f'(0, {inte[0]})'].shape == (3,3):
            Zdif_squared[f'({inte[0]}, {inte[1]})'] = (np.abs((zhat @ imap[f'(0, {inte[0]})'])[2] - \
                                                 (zhat @ imap[f'(0, {inte[1]})'])[2]))**2
        elif imap[f'(0, {inte[0]})'].shape != (3,3):
            Zdif_squared[f'({inte[0]}, {inte[1]})'] = (np.abs((imap[f'(0, {inte[0]})']) - \
                                                 (imap[f'(0, {inte[1]})'])))**2
            
    kappa_squared = {}
    kappa_prefactor = (16/3)*I*(I+1)

    for inte in intmap:
        bath_spins = [j for j in spin_dict if j != inte[0] and j != inte[1] and j != 0]
        summation = 0
    
        for spin in bath_spins:
            keys = []
            difference = 0
            for j in inte:
                if j > spin:
                    keys.append(f'({spin}, {j})')
                elif j < spin:
                    keys.append(f'({j}, {spin})')
            if imap[keys[0]].shape == (3,3):
                difference = ((zhat@imap[keys[0]][2]) - (zhat@imap[keys[1]][2]))**2
            elif imap[keys[0]].shape != (3,3):
                difference = (imap[keys[0]] - imap[keys[1]])**2
            summation += difference
    
        kappa_squared[f'({inte[0]}, {inte[1]})'] = summation*kappa_prefactor
    
    kappa_squared_keys = list(kappa_squared.keys())
    kappa_squared_values = list(kappa_squared.values())

    kappa = {key:np.sqrt(value) for key, value in zip(kappa_squared_keys,kappa_squared_values)}
    
    ff_rates = {}
    bnms = {}

    for pair in intmap:
        key = f'({pair[0]}, {pair[1]})'
        
        if imap[key].shape == (3,3):
            if sdb:
                ff_rates[key] = prefactor * (((zhat@imap[key][2])**2)/kappa[key])*np.exp(-(Zdif_squared[key])/(8*kappa_squared[key]))
                bnms[key] = (imap[key])
            
            elif not sdb:
                ff_rates[key] = prefactor * (((zhat@imap[key][2])**2)/kappa[key])*np.exp(-(0)/(8*kappa_squared[key]))
                bnms[key] = (imap[key])
            
        elif imap[key].shape != (3,3):
            if sdb:
                ff_rates[key] = prefactor * (((imap[key])**2)/kappa[key])*np.exp(-(Zdif_squared[key])/(8*kappa_squared[key]))
                bnms[key] = (imap[key])
                
            elif not sdb:
                ff_rates[key] = prefactor * (((imap[key])**2)/kappa[key])*np.exp(-(0)/(8*kappa_squared[key]))
                bnms[key] = (imap[key])        
            
    return ff_rates

def get_stddevFFrates(avg_xyzs,avg_hfs,cPos,numSpins,delta):
    
    if not isinstance(delta,bool):
        raise ValueError('Enter True or False for delta.')
    
    slist = list(range(1,numSpins))
    combs = combinations(slist,2)
    
    key_list = {}
    for comb in combs:
        key_list[f'{comb}'] = []
    
    for geom, hf in zip(avg_xyzs,avg_hfs):
    
        dim, spins, spin_types = sd.spin_dict_from_xyzs(cPos, avg_xyzs[geom])
    
        new_keys = []
        new_values = []
        for i, key in enumerate(list(avg_hfs[geom].keys())):
            new_keys.append(i + 1)
            new_values.append(avg_hfs[geom][key])
        new_dict = {key:value for key, value in zip(new_keys,new_values)}
    
        imap = sd.get_imap(numSpins,spins,new_dict)
    
        ff_rates = ff_rates_from_Sousa2003(imap,spins,sdb=delta)
        for key in key_list:
            key_list[key].append(ff_rates[key])

    stddevs = {}
    
    for key in key_list:
        mu_, std_ = norm.fit(key_list[key])
        stddevs[key] = std_
        
    return stddevs

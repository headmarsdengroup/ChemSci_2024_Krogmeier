import os, os.path
import numpy as np
from dependencies import *

class ME_CCE_2nd_order:
    def __init__(self,numSpins,hypTensors,equilGeom):
        
        if isinstance(numSpins,int):
            self.numSpins = numSpins
        else:
            print('Enter the integer number of spins in the system.')
        
        try:
            self.hypTensors = HFfromORCA(hypTensors)
        except:
            print('Input must be an ORCA output file.')
        
        try:
            self.equilGeom = getCoords(equilGeom,self.hypTensors[0])
        except:
            print('Input must be a .xyz file that matches the corresponding ORCA output file.')
            
    def gen_spinDict(self):
        
        self.spinDict = spin_dict_from_xyzs(self.equilGeom[list(self.equilGeom.keys())[0]],self.equilGeom)
    
    def gen_imap(self):
        
        self.imap = get_imap(self.numSpins,self.spinDict[1],self.hypTensors[1])
    
    def get_ffrates(self,dir_name,delta):
        
        soi = list(set(self.spinDict[-1][1:]))
        
        xyzs, hfs = avgeProcess(soi,dir_name)
        
        self.ffrates = get_stddevFFrates(xyzs[dir_name],hfs[dir_name],self.equilGeom[list(self.equilGeom.keys())[0]],self.numSpins,delta)
        
    def me_cce_2ndOrder(self,Bext,time_space):
        
        self.coherence = cce_2nd_order(self.imap,self.numSpins,self.spinDict[-1],self.spinDict[1],Bext,time_space,self.ffrates)

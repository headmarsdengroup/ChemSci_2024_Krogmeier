import numpy as np
from main import *

Bext = np.array([0,0,1000])
timesCu = np.linspace(0,0.2,500)
timesVO = np.linspace(0,0.08,500)

V1 = ME_CCE_2nd_order(14,'ORCA_output/V1.log','ORCA_output/V1.xyz')
V1.gen_spinDict()
V1.gen_imap()
V1.get_ffrates('ORCA_output/V1',True)
V1.me_cce_2ndOrder(Bext,timesVO)
np.save('V1coh.npy',V1.coherence)

V1d0 = ME_CCE_2nd_order(14,'ORCA_output/V1.log','ORCA_output/V1.xyz')
V1d0.gen_spinDict()
V1d0.gen_imap()
V1d0.get_ffrates('ORCA_output/V1',False)
V1d0.me_cce_2ndOrder(Bext,timesVO)
np.save('V1d0coh.npy',V1d0.coherence)

print('2 out of 12 done')

V2 = ME_CCE_2nd_order(14,'ORCA_output/V2.log','ORCA_output/V2.xyz')
V2.gen_spinDict()
V2.gen_imap()
V2.get_ffrates('ORCA_output/V2',True)
V2.me_cce_2ndOrder(Bext,timesVO)
np.save('V2coh.npy',V2.coherence)

V2d0 = ME_CCE_2nd_order(14,'ORCA_output/V2.log','ORCA_output/V2.xyz')
V2d0.gen_spinDict()
V2d0.gen_imap()
V2d0.get_ffrates('ORCA_output/V2',False)
V2d0.me_cce_2ndOrder(Bext,timesVO)
np.save('V2d0coh.npy',V2d0.coherence)

print('4 out of 12 done')

V3 = ME_CCE_2nd_order(14,'ORCA_output/V3.log','ORCA_output/V3.xyz')
V3.gen_spinDict()
V3.gen_imap()
V3.get_ffrates('ORCA_output/V3',True)
V3.me_cce_2ndOrder(Bext,timesVO)
np.save('V3coh.npy',V3.coherence)

V3d0 = ME_CCE_2nd_order(14,'ORCA_output/V3.log','ORCA_output/V3.xyz')
V3d0.gen_spinDict()
V3d0.gen_imap()
V3d0.get_ffrates('ORCA_output/V3',False)
V3d0.me_cce_2ndOrder(Bext,timesVO)
np.save('V3d0coh.npy',V3d0.coherence)

print('6 out of 12 done')

V4 = ME_CCE_2nd_order(14,'ORCA_output/V4.log','ORCA_output/V4.xyz')
V4.gen_spinDict()
V4.gen_imap()
V4.get_ffrates('ORCA_output/V4',True)
V4.me_cce_2ndOrder(Bext,timesVO)
np.save('V4coh.npy',V4.coherence)

V4d0 = ME_CCE_2nd_order(14,'ORCA_output/V4.log','ORCA_output/V4.xyz')
V4d0.gen_spinDict()
V4d0.gen_imap()
V4d0.get_ffrates('ORCA_output/V4',False)
V4d0.me_cce_2ndOrder(Bext,timesVO)
np.save('V4d0coh.npy',V4d0.coherence)

print('8 out of 12 done')

CuS = ME_CCE_2nd_order(10,'ORCA_output/CuS.log','ORCA_output/CuS.xyz')
CuS.gen_spinDict()
CuS.gen_imap()
CuS.get_ffrates('ORCA_output/CuS',True)
CuS.me_cce_2ndOrder(Bext,timesCu)
np.save('CuScoh.npy',CuS.coherence)

CuSd0 = ME_CCE_2nd_order(10,'ORCA_output/CuS.log','ORCA_output/CuS.xyz')
CuSd0.gen_spinDict()
CuSd0.gen_imap()
CuSd0.get_ffrates('ORCA_output/CuS',False)
CuSd0.me_cce_2ndOrder(Bext,timesCu)
np.save('CuSd0coh.npy',CuSd0.coherence)

print('10 out of 12 done')

CuSe = ME_CCE_2nd_order(10,'ORCA_output/CuSe.log','ORCA_output/CuSe.xyz')
CuSe.gen_spinDict()
CuSe.gen_imap()
CuSe.get_ffrates('ORCA_output/CuSe',True)
CuSe.me_cce_2ndOrder(Bext,timesCu)
np.save('CuSecoh.npy',CuSe.coherence)

CuSed0 = ME_CCE_2nd_order(10,'ORCA_output/CuSe.log','ORCA_output/CuSe.xyz')
CuSed0.gen_spinDict()
CuSed0.gen_imap()
CuSed0.get_ffrates('ORCA_output/CuSe',False)
CuSed0.me_cce_2ndOrder(Bext,timesCu)
np.save('CuSed0coh.npy',CuSed0.coherence)

print('12 out of 12 done')

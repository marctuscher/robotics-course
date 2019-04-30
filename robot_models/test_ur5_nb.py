#%%
import sys
sys.path.append("rai/rai/ry")
import libry as ry
from practical.objectives import moveToPosition

#%%
C = ry.Config()
V = C.view()
C.addFile('robot_models/ur5.g')
#%%
B = C.operate('')
C.makeObjectsConvex()
#%%
q = C.getJointState()

#%%
q[1] = 10

#%%
B.move([q], [10.0], False)

#%%
IK = C.komo_IK()

#%%
C.getFrameNames()

#%%

import sys
sys.path.append("rai/rai/ry")
import libry as ry
import numpy as np
from pdb import set_trace
from practical.objectives import moveToPosition, align

class RaiRobot():
    
    def __init__(self, nodeName:str, fileName:str):
        self.C = ry.Config()
        self.D = self.C.view()
        self.C.clear()
        self.C.addFile(fileName)
        self.q_home = self.C.getJointState()
        self.q_zero = self.q_home * 0.
        self.B = self.C.operate(nodeName)
        self.B.sync(self.C)
        self.real = False
        self.B.sendToReal(self.real)
        self.IK = self.C.komo_IK()
        self.C.addObject(name="ball", shape=ry.ST.sphere, size=[.1], pos=[.8,.8,1.5], color=[1,1,0])
    
    def getFrameNames(self)-> list:
        return self.C.getFrameNames()
    
    def moveToPosition(self, pos:np.ndarray, frameName:str):
        self.inverseKinematics([moveToPosition(pos, frameName)])
        
    def align(self, frameNames:list):
        self.inverseKinematics([align(frameNames)])
    
    def inverseKinematics(self, objectives:list):
        self.IK.clearObjectives()
        for obj in objectives:
            self.IK.addObjective(**obj)
        self.IK.optimize()
        self.C.setFrameState(self.IK.getConfiguration(0))
        self.B.move([self.C.getJointState()], [10.], False)
    
    def goHome(self):
        self.B.move([self.q_home], [10.0], False)
    
    def sendToReal(self, val:bool):
        self.real = val
        self.B.sendToReal(val)
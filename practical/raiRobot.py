import sys
sys.path.append("rai/rai/ry")
import libry as ry
import numpy as np
from pdb import set_trace
from practical.objectives import gazeAt, distance, scalarProductXZ, scalarProductZZ
import time 

class RaiRobot():
    
    def __init__(self, nodeName:str, fileName:str):
        self.C = ry.Config()
        self.views = {}
        self.views['default'] = self.C.view()
        self.C.clear()
        self.C.addFile(fileName)
        self.q_home = self.C.getJointState()
        self.q_zero = self.q_home * 0.
        self.B = self.C.operate(nodeName)
        self.B.sync(self.C)
        self.real = False
        self.B.sendToReal(self.real)
        self.C.makeObjectsConvex()
        self.C.addObject(name="ball", shape=ry.ST.sphere, size=[.05], pos=[0.55,.0,1.], color=[1.,1.,0.])
        self.C.addObject(name="ball2", shape=ry.ST.sphere, size=[.05], pos=[0.8,0,1], color=[1.,0.,0.])
    

    def getFrameNames(self)-> list:
        return self.C.getFrameNames()
    
    def inverseKinematics(self, objectives:list):
        """
        Calculate inverse kinematics by solving a constraint optimization problem, 
        given by objectives. 

        Using a new IK object, to ensure that all frames that have been added to 
        the configuration are also added to the computational graph of the solver.
        """
        IK = self.C.komo_IK()
        for obj in objectives:
            IK.addObjective(**obj)
        IK.optimize()
        self.C.setFrameState(IK.getConfiguration(0))
        self.move([self.C.getJointState()])
        self.B.wait()
    
    def goHome(self):
        self.B.move([self.q_home], [10.0], False)
        self.B.wait()
    
    def sendToReal(self, val:bool):
        self.real = val
        self.B.sendToReal(val)

    def setGripper(self, val:float, gripperIndex:int):
        """
        Directly set a value to the gripper joints.
        PR2: 
        - val [0,1]
        - gripperIndex:
         - leftGripper: -3
         - rightGripper: -4 
        """
        q = self.C.getJointState()
        q[gripperIndex] = val
        self.C.setJointState(q)
        self.move([q])

    
    def move(self, q:list):
        self.B.move(q, [10.0], False)

    def addCamera(self, name:str, parent:str, args:str, width:int, height:int, view:bool=False):
        self.C.addFrame(name=name, parent=parent, args=args)
        camView = self.C.cameraView()
        camView.addSensor(name=name + 'Sensor', frameAttached=name, width=width, height=height)
        if view:
            self.addView(name)
        return camView

        
    def addView(self, frameName:str):
        self.views[frameName] = self.C.view(frameName)

    
    def deleteFrame(self, frameName:str):
        assert(frameName in self.getFrameNames())
        if self.views[frameName]:
            self.views[frameName] = 0
            del self.views[frameName]
        self.C.delFrame(frameName)

    def deleteView(self, frameName:str):
        assert(self.views[frameName])
        self.views[frameName] = 0
        del self.views[frameName]
    

    def getPose(self, frame_name):
        pose = self.C.getFrameState(frame_name)
        return pose
        
    def grasp_ball(self, gripperFrame:str, targetFrame:str, gripperIndex:int):
        self.setGripper(0.04, gripperIndex)
        self.inverseKinematics(
            [
            gazeAt([gripperFrame, targetFrame]), 
            scalarProductXZ([gripperFrame, targetFrame], 0), 
            scalarProductZZ([gripperFrame, targetFrame], 0), 
            distance([gripperFrame, targetFrame], 0.1)
            ]
        )
        time.sleep(5)
        self.setGripper(0, gripperIndex)

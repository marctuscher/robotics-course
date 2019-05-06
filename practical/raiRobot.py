import sys
sys.path.append("rai/rai/ry")
import libry as ry
import numpy as np
from pdb import set_trace
from practical.objectives import gazeAt, distance, scalarProductXZ, scalarProductZZ, moveToPosition
import time 
from scipy.spatial.transform import Rotation as R

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
        #self.C.addObject(name="ball", shape=ry.ST.sphere, size=[.05], pos=[0.55,.0,1.], color=[1.,1.,0.])
        self.C.addObject(name="ball2", shape=ry.ST.sphere, size=[.05], pos=[0.8,0,1], color=[0.,1.,0.])
    

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
        return self.C.getJointState()


    def path(self, objectives):
        path = self.C.komo_path(1, 100, 5)
        for obj in objectives:
            path.addObjective(**obj)
        path.optimize()
        self.C.setFrameState(path.getConfiguration(0))
        return self.C.getJointState()

    def goHome(self):
        self.move([self.q_home])
    
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
        self.B.move(q, [2.0], False)
        #self.B.wait()

    def getCamView(self, view:bool, **kwargs):
        if view:
            self.addView(kwargs['frameAttached'])
        camView = self.C.cameraView()
        camView.addSensor(**kwargs)
        camView.selectSensor(kwargs['name'])
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
        # a pose is represented 7D vector (x,y,z, qw,qx,qy,qz)
        pose = self.C.getFrameState(frame_name)
        return pose
        
    def grasp(self, gripperFrame:str, targetFrame:str, gripperIndex:int):
        self.setGripper(0.04, gripperIndex)
        q = self.inverseKinematics(
            [
            gazeAt([gripperFrame, targetFrame]), 
            scalarProductXZ([gripperFrame, targetFrame], 0), 
            scalarProductZZ([gripperFrame, targetFrame], 0), 
            distance([gripperFrame, targetFrame], 0.1)
            ]
        )
        self.move([q])
        self.setGripper(0, gripperIndex)
    
    def trackAndGraspTarget(self, targetPos, targetFrame, gripperFrame, gripperIndex, gripperOpenVal):
        target = self.C.frame(targetFrame)
        #self.setGripper(gripperOpenVal, gripperIndex)
        if targetPos:
            target.setPosition(targetPos)
            q = self.path(
                [
                    gazeAt([gripperFrame, targetFrame]), 
                    scalarProductXZ([gripperFrame, targetFrame], 0), 
                    scalarProductZZ([gripperFrame, targetFrame], 0), 
                    distance([gripperFrame, targetFrame], -0.1)
                ]
            )
            self.move([q])


    def computeCartesianPos(self, framePos, frameName):

        # get the pose of the desired frame in respect to world coordinates
        pose = robot.C.getFrameState(frameName)
        pos = pose[0:3]
        rot = pose[3:7]

        # we transform a vector v using a normalized quaternion q, where q' is the complex conjugated quaternion
        # p_ = q * v * q'
        v = np.concatenate((np.array([0.]), framePos), axis=0)
        v_ = robot.quat_multiply(robot.quat_multiply(rot,v), robot.quat_conj(rot))
        v_ = v_[1:4]

        return v_ + pos




    def quat_multiply(self, quat0, quat1):
        # as numpy lacks a proper quaternion multiplication method, we provide it here
        w0, x0, y0, z0 = quat0
        w1, x1, y1, z1 = quat1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0, 
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    def quat_conj(self, quat0):
        w0, x0, y0, z0 = quat0
        return np.array([w0, -x0, -y0, -z0], dtype=np.float64)
    

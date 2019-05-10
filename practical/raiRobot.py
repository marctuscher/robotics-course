import sys
sys.path.append("rai/rai/ry")
import libry as ry
import numpy as np
import quaternion
from practical import utils
from pdb import set_trace
from practical.objectives import gazeAt, distance, scalarProductXZ, scalarProductZZ, moveToPosition, moveToPose
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
        self.C.addObject(name="ball2", shape=ry.ST.sphere, size=[.05], pos=[0.8,0,1], color=[0.,1.,0.])
        # add camera on baxter head -> pcl is the camera frame
        self.pcl = self.C.addFrame('pcl', 'head')
        self.pcl.setRelativePose('d(-90 0 0 1) t(-.08 .205 .115) d(26 1 0 0) d(-1 0 1 0) d(6 0 0 1) ')
        self.camView = self.getCamView(True, frameAttached='pcl', name='headCam', width=640, height=480, focalLength=580./480., orthoAbsHeight=-1., zRange=[.1, 50.], backgroundImageFile='')
        if nodeName:
            self.cam = ry.Camera(nodeName, "/camera/rgb/image_rect_color", "/camera/depth_registered/image_raw")


    def getFrameNames(self)-> list:
        return self.C.getFrameNames()
    
    def inverseKinematics(self, objectives:list):
        """
        Calculate inverse kinematics by solving a constraint optimization problem, 
        given by objectives. 

        Using a new IK object, to ensure that all frames that have been added to 
        the configuration are also added to the computational graph of the solver.
        """
        IK = self.C.komo_IK(False)
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
        pose = self.C.getFrameState(frameName)
        pos = pose[0:3]
        rot = pose[3:7]
        q = utils.arr2quat(rot)

        homTF_pose = utils.pose7d2homTF(pose)
        homTF_framePose = utils.pose7d2homTF(np.concatenate(framePos, np.array([1, 0, 0, 0]), axis=0))
        homTF_res = homTF_pose @ homTF_framePose
        t = np.transpose(homTF_res[0:3, 3])

        return t

        # we transform a vector v using a normalized quaternion q, where q' is the complex conjugated quaternion
        # p_ = q * v * q'
        v = np.concatenate((np.array([0.]), framePos), axis=0)
        v = utils.arr2quat(v)
        #v_ = quatMultiply(quatMultiply(quatConj(rot),v),rot)
        v_ = q * v * np.invert(q)
        v_ = v_[1:4]
        return v_ + pos
 

    def imgAndDepth(self):
        assert self.cam
        return self.cam.getRgb(), self.cam.getDepth()

    def virtImgAndDepth(self):
        self.camView.updateConfig(self.C)
        return self.camView.computeImageAndDepth()

    def computeCartesianTwist(self, actPose, desPose, gain):
        # actPose and des Pose must be in the same reference frame!     

        t_err = desPose[0:3] - actPose[0:3]

        q_act = utils.arr2quat(actPose[3:7])
        q_act_inv = np.invert(q_act)
        q_des = utils.arr2quat(desPose[3:7])

        # Compute rotational error in quaternions
        # q_err = q_soll * (q_ist)^-1
        q_err = q_des * q_act_inv
        q_err = utils.quat2arr(q_err)

        ''' twist with angular euluer twist
        # numpy quaternion uses a zyx-euler convention
        R = quaternion.as_rotation_matrix(q_err)
        zyx = utils.rotm2eulZYX(R)

        twist = np.array([t_err[0], t_err[1], t_err[2], zyx[2], zyx[1], zyx[0]])
        '''

        twist = np.array([t_err[0], t_err[1], t_err[2], q_err[0], q_err[1], q_err[2], q_err[3]])

        # calculate the error in euler angles as this represents the angular rotatory twists
        # TODO: tune this value 

        twist = twist * gain     

        print(twist)

        return twist


    def sendCartesianTwist(self, twist, frameName):
        '''
        This methods sends cartesian twists in respect to a given frame
        to the robot. A twist is a 6D vector containing a linear and an 
        angular twist
        '''

        #act_pose = getPose(frameName)
        return 0
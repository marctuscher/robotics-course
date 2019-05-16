import sys
sys.path.append("rai/rai/ry")
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
import libry as ry
import numpy as np
import quaternion
from practical import utils
from practical.vision import baxterCamIntrinsics
from pdb import set_trace
from practical.objectives import gazeAt, distance, scalarProductXZ, scalarProductZZ, moveToPosition, moveToPose, accumulatedCollisions, qItself, moveToPosition
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
        self.C.addObject(name="ball", shape=ry.ST.sphere, size=[.05], pos=[0.8,0,1], color=[0.,0.,1.])
        # add camera on baxter head -> pcl is the camera frame
        self.pcl = self.C.addFrame('pcl', 'head')
        self.pcl.setRelativePose('d(-90 0 0 1) t(-.08 .205 .115) d(26 1 0 0) d(-1 0 1 0) d(6 0 0 1) ')
        self.camView = self.getCamView(False, frameAttached='pcl', name='headCam', width=640, height=480, focalLength=580./480., orthoAbsHeight=-1., zRange=[.1, 50.], backgroundImageFile='')
        self.cam = None
        # add camera on baxters left hand -> lhc (left hand cam)
        self.lhc = self.C.addFrame('lhc', 'left_hand_camera')
        self.lhc.setRelativePose('t(-.0 .0 .0) d(-180 1 0 0) ')
        self.camView_lhc = self.getCamView(False, frameAttached='lhc', name='leftHandCam', width=640, height=480, focalLength=580./480., orthoAbsHeight=-1., zRange=[.1, 50.], backgroundImageFile='')
        self.cam_lhc = None

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
        IK.optimize(True)
        self.C.setFrameState(IK.getConfiguration(0))
        return self.C.getJointState()


    def path(self, objectives):
        path = self.C.komo_path(1, 100, 5)
        for obj in objectives:
            path.addObjective(**obj)
        path.optimize(False)
        self.C.setFrameState(path.getConfiguration(0))
        return self.C.getJointState()

    def goHome(self):
        self.move(self.q_home)
    
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
        BAX: 
        - val [0,1]
        - gripperIndex:
         - leftGripper: -1
         - rightGripper: -2 
         
        """
        q = self.C.getJointState()
        q[gripperIndex] = val
        self.C.setJointState(q)
        self.move([q])

    
    def move(self, q:list):
        self.B.moveHard(q)
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
        if frameName in self.views:
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
    
    def trackAndGraspTarget(self, targetPos, targetFrame, gripperFrame, sendQ=False):
        target = self.C.frame(targetFrame)
        target.setPosition(targetPos)
        q = self.C.getJointState()
        if sendQ:
            q = self.inverseKinematics(
                [
                    gazeAt([gripperFrame, targetFrame]), 
                    scalarProductXZ([gripperFrame, targetFrame], 0), 
                    scalarProductZZ([gripperFrame, targetFrame], 0), 
                    #distance([gripperFrame, targetFrame], -0.1),
                    #accumulatedCollisions(),
                    qItself(q, 0.1),
                    moveToPosition(targetPos, 'baxterR')

                ]
            )
            self.move(q)
        #TODO: dont do this
        #self.setGripper(0, -1) # close right gripper
        #self.setGripper(0, -2) # close left gripper


    def computeCartesianPos(self, framePos, frameName):

        # get the pose of the desired frame in respect to world coordinates
        pose = self.C.getFrameState(frameName)
        pos = pose[0:3]
        R = utils.quat2rotm(pose[3:7])
        return pos + R @ np.array(framePos)



    def imgAndDepth(self, camName, virtual=False):
        if camName == 'cam':
            if not self.cam or virtual:
                self.camView.updateConfig(self.C)
                img, d = self.camView.computeImageAndDepth()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img, d = self.cam.getRgb(), self.cam.getDepth()

        elif camName == 'cam_lhc':
            if not self.cam_lhc or virtual:
                self.camView_lhc.updateConfig(self.C)
                img, d = self.camView_lhc.computeImageAndDepth()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img, d = self.cam_lhc.getRgb(), self.cam_lhc.getDepth()
                
        
        return img, d


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

    def sync(self):
        self.B.sync(self.C)

    def addPointCloud(self):
        self.sync()
        self.pcl.setPointCloud(self.cam.getPoints([baxterCamIntrinsics['fx'],baxterCamIntrinsics['fy'],baxterCamIntrinsics['px'],baxterCamIntrinsics['py']]), self.cam.getRgb())
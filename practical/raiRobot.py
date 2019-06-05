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
from practical import vision
from pdb import set_trace
from practical.objectives import gazeAt, distance, scalarProductXZ, scalarProductZZ, moveToPosition, moveToPose, accumulatedCollisions, qItself, moveToPosition, scalarProductYZ, positionDiff
import time 


def syncBefore(f):
    def need_sync(*args, **kwargs):
        self = args[0]
        self._sync()
        res = f(*args, **kwargs)
        return res
    return need_sync

def syncAfter(f):
    def need_sync(*args, **kwargs):
        res = f(*args, **kwargs)
        self = args[0]
        self._sync()
        return res
    return need_sync

def syncAndReinitKomo(f):
    def need_sync(*args, **kwargs):
        res = f(*args, **kwargs)
        self = args[0]
        self._sync()
        self.path = self.C.komo_path(self.numPhases, self.stepsPerPhase, self.timePerPhase, True)
        self.IK = self.C.komo_IK(True)
        return res 
    return need_sync

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
        self.real = False
        self.B.sendToReal(self.real)
        self.C.makeObjectsConvex()
        #self.C.addObject(name="ball2", shape=ry.ST.sphere, size=[.05], pos=[0.8,0,1], color=[0.,1.,0.])
        #self.C.addObject(name="ball", shape=ry.ST.sphere, size=[.05], pos=[0.8,0,1], color=[0.,0.,1.])
        # add camera on baxter head -> pcl is the camera frame
        self.pcl = self.C.addFrame('pcl', 'head')
        self.pcl.setRelativePose('d(-90 0 0 1) t(-.08 .205 .115) d(26 1 0 0) d(-1 0 1 0) d(6 0 0 1) ')
        self.pcl.setPosition([-0.0472772, 0.226517, 1.79207 ])
        self.pcl.setQuaternion([0.969594, 0.24362, -0.00590741, 0.0223832])
        self.camView = self.getCamView(False, frameAttached='pcl', name='headCam', width=640, height=480, focalLength=580./480., orthoAbsHeight=-1., zRange=[.1, 50.], backgroundImageFile='')
        self.cam = None
        # add camera on baxters left hand -> lhc (left hand cam)
        self.lhc = self.C.addFrame('lhc', 'left_hand_camera')
        self.lhc.setRelativePose('t(-.0 .0 .0) d(-180 1 0 0) ')
        self.camView_lhc = self.getCamView(False, frameAttached='lhc', name='leftHandCam', width=640, height=480, focalLength=580./480., orthoAbsHeight=-1., zRange=[.1, 50.], backgroundImageFile='')
        self.cam_lhc = None
        self.IK = self.C.komo_IK(True)
        self.numPhases = 1
        self.stepsPerPhase = 30
        self.timePerPhase = 10
        self.path = self.C.komo_path(self.numPhases, self.stepsPerPhase, self.timePerPhase, False)
        self.B.sync(self.C)
        self.pathObjectives = []
        self.ikObjectives = []
        if nodeName:
            self.cam = ry.Camera(nodeName,"/camera/rgb/image_rect_color", "/camera/depth_registered/image_raw")
        
    def __delete__(self, instance):
        self.C = 0
        self.B = 0
        for key in self.views.keys():
            self.views[key] = 0
        if self.cam:
            self.cam = 0

    def getFrameNames(self)-> list:
        return self.C.getFrameNames()


###################### Motion Planning #################################

    def addIkObjectives(self, objectives):
        if False: #objectives == self.ikObjectives:
            return
        else:
            self.ikObjectives = []
            self.IK.clearObjectives()
            for obj in objectives:
                self.ikObjectives.append(obj)
                self.IK.addObjective(**obj)

    @syncBefore   
    def optimizeIk(self):
        self.IK.optimize(True)
        q_curr = self.C.getJointState()
        self.C.setFrameState(self.IK.getConfiguration(0))
        q = self.C.getJointState()
        q[-1] = q_curr[-1]
        q[-2] = q_curr[-2]
        return q


    def addPathObjectives(self, objectives, clear=False):
        self.path.setConfigurations(self.C)
        if False: #objectives == self.pathObjectives:
            return
        else:
            self.path.clearObjectives()
            self.pathObjectives = []
            for obj in objectives:
                self.pathObjectives.append(obj)
                self.path.addObjective(**obj)


    @syncBefore   
    def optimizePath(self):
        self.path.optimize(True)
        q = []
        t = self.path.getT()
        for i in range(t):
            self.C.setFrameState(self.path.getConfiguration(i))
            q_tmp = self.C.getJointState()
            q_tmp[-1] = 0
            q_tmp[-2] = 0
            q += [q_tmp]
        return q


###################### Motion #################################

    def goHome(self, hard=True ,randomHome=False):
        if randomHome and not self.real:
            q = self.q_home.copy()
            noise = np.random.normal(0, 0.1, q.shape[0]-2)
            q[0:-2] += noise
        else:
            q = self.q_home
        self.move(q, hard)

    @syncAfter
    def move(self, q:list, hard=True):
        if hard:
           self.B.moveHard(q)
        else:
            self.movePath([q])
        self.C.setJointState(q)

    def movePath(self, path):
        self.B.move(path, [5/30 * i for i in range(len(path))], True)
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
        BAX: 
        - val [0,1]
        - gripperIndex:
         - leftGripper: -1
         - rightGripper: -2 
         
        """
        q = self.C.getJointState()
        q[gripperIndex] = val
        self.move(q)




###################### Vision #################################


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
    
    def getCamView(self, view:bool, **kwargs):
        if view:
            self.addView(kwargs['frameAttached'])
        camView = self.C.cameraView()
        camView.addSensor(**kwargs)
        camView.selectSensor(kwargs['name'])
        return camView
        
    def addView(self, frameName:str):
        self.views[frameName] = self.C.view(frameName)

    @syncAfter
    def addPointCloud(self):
        self.pcl.setPointCloud(self.cam.getPoints([vision.baxterCamIntrinsics['fx'],vision.baxterCamIntrinsics['fy'],vision.baxterCamIntrinsics['cx'],vision.baxterCamIntrinsics['cy']]), self.cam.getRgb())
    
###################### Utilities #################################

    @syncAndReinitKomo
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

    def _sync(self):
        self.B.sync(self.C)
        self.path.setConfigurations(self.C)
        self.IK.setConfigurations(self.C)

    def _checkTarget(self, targetFrame):
        if targetFrame not in self.C.getFrameNames():
            print('adding')
            self.addObject(name=targetFrame, shape=ry.ST.sphere, size=[.01], pos=[0,0,0], color=[0.,0.,1.])
        return self.C.frame(targetFrame)

    @syncAndReinitKomo    
    def addObject(self, **kwargs):
        return self.C.addObject(**kwargs)


    def _calcPathSpeed(self, speed, pathLen):
        secs = []
        for i in range(pathLen):
            if i > pathLen / 2:
                secs.append(speed / pathLen *3)
            else:
                secs.append(3 * speed / pathLen)
        return secs


###################### Predefined stuff #################################
    
    def trackAndGraspTarget(self, targetPos, targetFrame, gripperFrame, sendQ=False):
        target = self._checkTarget(targetFrame)
        target.setPosition(targetPos)
        q = self.C.getJointState()
        if sendQ:
            self.addIkObjectives(
                [
                    #gazeAt([gripperFrame, targetFrame]), 
                    scalarProductYZ([gripperFrame, targetFrame], 0), 
                    scalarProductZZ([gripperFrame, targetFrame], 0), 
                    #distance([gripperFrame, targetFrame], -0.1),
                    moveToPosition(targetPos, gripperFrame)
                    #positionDiff([targetFrame, gripperFrame], 0, 1)

                ]
            )
            q = self.optimizeIk()
            self.move(q)


    def trackPath(self, targetPos, targetFrame, gripperFrame, sendQ=False):
        target = self._checkTarget(targetFrame)
        target.setPosition(targetPos)
        q = self.C.getJointState()
        if sendQ:
            self.addPathObjectives(
                [
                    #gazeAt([gripperFrame, targetFrame]), 
                    scalarProductYZ([gripperFrame, targetFrame], 0), 
                    scalarProductZZ([gripperFrame, targetFrame], 1), 
                    #distance([gripperFrame, targetFrame], -0.1),
                    #accumulatedCollisions(1),
                    #qItself(self.q_home, 0.05),
                    #positionDiff([targetFrame, gripperFrame], 0, 1)
                    moveToPosition(targetPos, gripperFrame)
                ]
            )
            q = self.optimizePath()
            self.movePath(q)
            return q

    def graspPath(self, targetPos, angle,targetFrame, gripperFrame, sendQ=False, collectData=False):
        target = self._checkTarget(targetFrame)
        rotM = utils.rotz(angle)
        quat =utils.rotm2quat(rotM)
        target.setPosition(targetPos)
        target.setQuaternion(quat)
        approachEnd = 0.7
        q = self.C.getJointState()
        if sendQ:
            self.addPathObjectives(
                [
                    #gazeAt([gripperFrame, targetFrame]), 
                    scalarProductYZ([gripperFrame, targetFrame], 0), 
                    scalarProductZZ([gripperFrame, targetFrame], 1), 
                    #distance([gripperFrame, targetFrame], -0.1),
                    #accumulatedCollisions(1),
                    qItself(self.q_home, 0.01, [1.]),
                    #positionDiff([targetFrame, gripperFrame], 0, 1)
                    moveToPosition([targetPos[0], targetPos[1], targetPos[2] + 0.2], gripperFrame, [0, 0.7]),
                    moveToPosition([targetPos[0], targetPos[1], targetPos[2]], gripperFrame, [0.7, 1.0])
                ]
            )
            q = self.optimizePath()
            #self.path.display()
            print(self.path.getReport())
            self.movePath(q)
            return q
    


    def approachGrasp(self, grasp, d, gripperFrame='baxterR', targetFrame='graspTarget'):
        res =  vision.getGraspPosition(d, grasp['x'], grasp['y'])
        if res:
            pc, _, _ = res
            pos = self.computeCartesianPos(pc, 'pcl')
            self.graspPath(np.array([pos[0], pos[1], pos[2]]), grasp['angle'],targetFrame, gripperFrame, sendQ=True)



###################### Coordinate Transformations #################################

    def computeCartesianPos(self, framePos, frameName):
        pose = self.C.getFrameState(frameName)
        pos = pose[0:3]
        R = utils.quat2rotm(pose[3:7])
        print(R)
        print(pos)
        return pos + R @ np.array(framePos)



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


        twist = np.array([t_err[0], t_err[1], t_err[2], q_err[0], q_err[1], q_err[2], q_err[3]])

        # calculate the error in euler angles as this represents the angular rotatory twists
        # TODO: tune this value 

        twist = twist * gain     

        print(twist)

        return twist



###################### Baxter Stuff #################################

    def closeBaxterR(self):
        self.setGripper(0.07, -2)
    
    def openBaxterR(self):
        self.setGripper(0.0, -2)

    def closeBaxterL(self):
        self.setGripper(0.07, -1)
    
    def openBaxterL(self):
        self.setGripper(0.0, -1)


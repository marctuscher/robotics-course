import sys
sys.path.append("rai/rai/ry")
import libry as ry


def moveToPosition(pos:list, frame:str):
    """
    Move frame to a given position
    """
    return {
        'type': ry.OT.eq,
        'feature': ry.FS.position, 
        'frames': [frame],
        'target': pos,
    }

def moveToPose(pos:list, frame:str):
    """
    Move frame to a given position
    """
    return {
        'type': ry.OT.eq,
        'feature': ry.FS.pose, 
        'frames': [frame],
        'target': pos,
    }

def gazeAt(frames:list):
    """
    Gaze at a frame
    """
    return {
        'type': ry.OT.eq,
        'feature': ry.FS.gazeAt, 
        'frames': frames,
        'target': [0, 0],
    }

def scalarProductXZ(frames:list, target):
    """
    Align two frames. The scalar product of two of the axes of those frames is
    equal to 1, if aligned.
    """
    return {
        'type': ry.OT.eq,
        'feature': ry.FS.scalarProductXZ,
        'frames': frames,
        'target': [target]
    }

def scalarProductZZ(frames:list, target):
    """
    Align two frames. The scalar product of two of the axes of those frames is
    equal to 1, if aligned.
    """
    return {
        'type': ry.OT.eq,
        'feature': ry.FS.scalarProductZZ,
        'frames': frames,
        'target': [target]
    }

def distance(frames:list, target:float):
    """
    Minimize the distance between two given frames.

    Parameters:
     - frames: list of frames
     - target: target distance between frames, should be <= 0
        - 0: positions of frames align
        - -0.1: positions of frames are 0.1 apart
    """
    return {
        'type': ry.OT.eq,
        'feature': ry.FS.distance,
        'frames': frames,
        'target': [target]    
        }

def qItself():
    """
    Minimize zhe configuration joint vector.

    Parameters:
     - 
    """
    return {
        'type': ry.OT.eq,
        'feature': ry.FS.qItself,
        'target': [0],   
        }


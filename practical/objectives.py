import sys
sys.path.append("rai/rai/ry")
import libry as ry


def moveToPosition(pos, frame):
    return {
        'type': ry.OT.none,
        'feature': ry.FS.position, 
        'frames': [frame],
        'target': pos,
    }

def align(frames):
    return {
        'type': ry.OT.none,
        'feature': ry.FS.gazeAt,
        'frames': frames,
        'target': [1]
    }
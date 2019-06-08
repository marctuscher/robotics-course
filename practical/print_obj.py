import sys
sys.path.append(".")
gripperFrame = "baxterR"
targetFrame = "ball"
import ry.libry as ry
l = [
                    {'type': ry.OT.eq, 'feature': ry.FS.scalarProductYZ, 'frames': [gripperFrame, targetFrame], 'target': [0], 'time': []},
                    {'type': ry.OT.eq, 'feature': ry.FS.scalarProductZZ, 'frames': [gripperFrame, targetFrame], 'target': [1], 'time': []},
                    #{'type': ry.OT.sos, 'feature': ry.FS.qItself, 'frames': [], 'target': self.q_home, 'time': [1.]},
                    {'type': ry.OT.eq, 'feature': ry.FS.distance, 'frames': [gripperFrame, targetFrame], 'target': [-0.2], 'time': [.8]},
                    {'type': ry.OT.eq, 'feature': ry.FS.positionDiff, 'frames': [gripperFrame, targetFrame], 'time': [1.]},
                    {'type': ry.OT.eq, 'feature': ry.FS.qItself, 'frames': [], 'order': 1,  'time': [1.]},
]

for obj in l:
    print(', '.join(['{}={!r}'.format(k, v) for k, v in obj.items()]))
        


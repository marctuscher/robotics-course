#%%
import os
os.getcwd()
#%%
import sys
sys.path.append('ry')
from libry import *
from numpy import *

#%% [markdown]
# ## Setting up a basic Config
# 
# The starting point is to create a `Configuration`.

#%%
K = Config()
D = K.view()


#%%
K.addFile('../rai-robotModels/baxter/baxter-clean.g');

#%% [markdown]
# Note that the view was updated automatically. (Internally, the view 'subscribes' to updates of the shared memory of Config).

#%%
K.addObject(name="ball", shape=ST.sphere, size=[.1], pos=[.8,.8,1.5], color=[1,1,0])

#%% [markdown]
# One can also add convex meshes (just passing the passing the vertex array), or use sphere-swept convex meshes (ssBox, capsule, sphere, etc)

#%%
K.addObject(name="hand", parent="pr2L", shape=ST.ssBox, size=[.2,.2,.1,.02], pos=[0,0,-.1], color=[1,1,0])

#%% [markdown]
# In this last example, the new object has another frame (pr2L) as *parent*. This means that it is permanently attached to this parent. pos and quat/rot are interpreted relative to the parent.
#%% [markdown]
# We can translate the framenames to attributes, so that they can be accessed by tab completion

#%%
class F(object):
    def __init__(self):
        print("bla")
    

for n in K.getFrameNames():
    setattr(F, n, n)

print(F.sink1)

#%% [markdown]
# ## Joint and Frame State
# 
# A configuration is a tree of n frames. Every frame has a pose (position & quaternion), which is represented as a 7D vector (x,y,z, qw,qx,qy,qz). The frame state is the n-times-7 matrix, where the i-th row is the pose of the i-th frame.
# 
# A configuration also defines joints, which means that the relative transfromation from a parent to a child frame is parameterized by degrees-of-freedom (DOF). If the configuration has in total n DOFs, the joint state is a n-dimensional vector.
# 
# Setting the joint state implies computing all relative transformations, and then forward chaining all transformations to compute all frame poses. So setting the joint state also sets the frame state.
#      
# Setting the frame state allows you to set frame poses that are inconsistent/impossible w.r.t. the joints! Setting the frame state implies computing all relative transformations from the frame poses, and then assigning the joint state to the *projection( onto the actual DOFs

#%%
q = K.getJointState()
print('joint state: ', q)
print('joint names: ', K.getJointNames() )

#%% [markdown]
# Let's move the configuration by adding to the joint configuration

#%%
q[2] = q[2] + 1.
K.setJointState(q)

#%% [markdown]
# The *frame state* is a $n\times 7$ matrix, which contains for all of $n$ frames the 7D pose. A pose is stored as [p_x, p_y, p_z, q_w, q_x, q_y, q_z], with position p and quaternion q.

#%%
X0 = K.getFrameState()
print('frame state: ', X0)

#%% [markdown]
# Let's do a questionable thing: adding .1 to all numbers in the pose matrix

#%%
X = X0 + .1
K.setFrameState(X)

#%% [markdown]
# The rows of X have non-normalized quaternions! These are normalized when setting the frame state.
# 
# Also, the frame poses are now *inconsistent* to the joint constraints! We can read out the projected joint state, set the joint state, and get a consistent state again:

#%%
K.setJointState( K.getJointState() )

#%% [markdown]
# Let's bring the configuration back into the state before the harsh *setFrame*

#%%
K.setFrameState(X0)

#%% [markdown]
#  ## Selecting joints
# 
# Often one would like to choose which joints are actually active, that is, which joints are referred to in q. This allows one to sub-selection joints and work only with projections of the full configuration state. This changes the joint state dimensionality, including ordering of entries in q.
# 
# However, the frame state is invariant against such selection of active joints.

#%%
K.selectJointsByTag(["armL","base"])
q = K.getJointState()
print('joint state: ', q)
print('joint names: ', K.getJointNames() )

#%% [markdown]
# ## Features & Jacobians
# 
# A core part of rai defines features over configurations. A feature is a differentiable mapping from a configuration (or set of configurations) to a vector. Starndard features are "position-of-endeffector-X" or "distance/penetration-between-convex-shapes-A-and-B", etc. But there are many, many more features defined in rai, like error of Newton-Euler-equations for an object, total energy of the system, etc. Defining differentiable is the core of many functionalities in the rai code.
# 
# Let's define a basic feature over C: the 3D (world coordinate) position of pr2L (left hand)

#%%
F = K.feature(FS.position, ["pr2L"])

#%% [markdown]
# We can now evaluate the feature, and also get the Jacobian:

#%%
print(F.description(K))

[y,J] = F.eval(K)
print('hand position =', y)
print('Jacobian =', J)

#%% [markdown]
# Another example

#%%
F2 = K.feature(FS.distance, ["hand", "ball"])
print(F2.description(K))


#%%
F2.eval(K)

#%% [markdown]
# When you call a feature on a *tuple* of configurations, by default it computes the difference, acceleration, etc, w.r.t. these configurations

#%%
C2 = Config()
C2.copy(K)  #this replicates the whole structure
V2 = C2.view()


#%%
F.eval((K,C2))[0]

#%% [markdown]
# This should be zero. To see a difference, let's move the 2nd configuration:

#%%
# just to see a difference between the two:
q = C2.getJointState()
q = q - .1
C2.setJointState(q)
y = F.eval((K,C2))[0]
print('hand difference (y(C2) - y(K)) =', y)

#%% [markdown]
# An acceleration example:

#%%
C3 = Config()
C3.copy(K);
C3.setJointState(q + .2);


#%%
(y,J) = F.eval((K, C2, C3))
print('hand acceleration = ', y)
print('shape of Jacobian =', J.shape)

#%% [markdown]
# Note that the Jacobian is now w.r.t. all three configurations! It is of size 3x3xdim(q). Let's retrieve the Jacobian w.r.t. C3 only:

#%%
J = J.reshape((3,3,q.size))
print('shape of Jacobian =', J.shape)
J[:,1,:]

#%% [markdown]
# Another example, when the dimensions of K and C2 are different:

#%%
C2.selectJointsByTag(['armL'])
(y,J) = F.eval((K,C2))
print('shape of Jacobian =', J.shape)
print('dimensions of configurations =', (K.getJointDimension(), C2.getJointDimension()))

#%% [markdown]
# Finally, we can linearly transform features by setting 'scale' and 'target':

#%%
#F.scale = 10.
#F.target = [0., 0., 1.];
#  y = F(C);
#  //.. now y = F.scale * (f(C) - F.target), which means y is zero if
#  //the feature f(C) is equal to the target (here, if the hand is in world
#  //position (0,0,1) )
#
#  //F.scale can also be a matrix, which can transform the feature also to lower dimensionality
#  F.scale = arr(1,3,{0., 0., 1.}); //defines the 1-times-3 matrix (0 0 1)
#  y = F(C);
#  //.. now y is 1-dimensional and captures only the z-position 

#%% [markdown]
# # THE REST IS PRELIM
#%% [markdown]
# We can also add a frame, attached to the head, which has no shape associated to it, but create a view is associated with that frame:

#%%
K.addFrame(name='camera', parent='head_tilt_link', args='Q:<d(-90 1 0 0) d(180 0 0 1)> focalLength:.3')
C = K.view(frame='camera')

#%% [markdown]
# When we move the robot, that view moves with it:

#%%
K.setJointState(q=asarray([1.]), joints=['head_pan_joint'])

#%% [markdown]
# To close a view (or destroy a handle to a computational module), we reassign it to zero. We can also remove a frame from the configuration.

#%%
C = 0
K.delFrame('camera')

#%% [markdown]
# This solves a simple IK problem, defined by an equality constraint on the difference in position of 'ball' and 'hand'

#%%
IK.getReport()

#%% [markdown]
# We can reuse the optimization object, change the objective a bit (now the position difference is constrained to be [.1,.1,.1] in world coordinates), and reoptimize

#%%
IK.clearObjectives()
IK.addObjective(type=OT.eq, feature=FS.positionDiff, frames=['hand', 'ball'], target=[.1, .1, .1])
IK.optimize()
K.setFrameState( IK.getConfiguration(0) )

#%% [markdown]
# TODO demos:
# 
# * rename Camera -> View
# * copy configurations
# * have multiple configurations and views in parallel
# * selecting/modifying DOFs (i.e., which joints are considered DOFs)
# * I/O with other file formats?

#%%
K.setFrameState(X0)


#%%
K=0


#%%
D=0


#%%



#%%




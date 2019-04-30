body base_link { mass:4.0 inertiaTensor:[0.00443333156 0.0 0.0 0.00443333156 0.0 0.0072] }
shape visual base_link_1 (base_link) {
   type:mesh mesh:'meshes/ur5/visual/Base.dae' color:[0.7 0.7 0.7 1.0] colorName:LightGrey  visual }
shape collision base_link_0 (base_link) {
    color:[.8 .2 .2 .5], type:mesh mesh:'meshes/ur5/collision/base.stl'  contact:-2 }
body shoulder_link { mass:3.7 inertiaTensor:[0.010267495893 0.0 0.0 0.010267495893 0.0 0.00666] }
shape visual shoulder_link_1 (shoulder_link) {
   type:mesh mesh:'meshes/ur5/visual/Shoulder.dae' color:[0.7 0.7 0.7 1.0] colorName:LightGrey  visual }
shape collision shoulder_link_0 (shoulder_link) {
    color:[.8 .2 .2 .5], type:mesh mesh:'meshes/ur5/collision/shoulder.stl'  contact:-2 }
body upper_arm_link { mass:8.393 inertiaTensor:[0.22689067591 0.0 0.0 0.22689067591 0.0 0.0151074] }
shape visual upper_arm_link_1 (upper_arm_link) {
   type:mesh mesh:'meshes/ur5/visual/UpperArm.dae' color:[0.7 0.7 0.7 1.0] colorName:LightGrey  visual }
shape collision upper_arm_link_0 (upper_arm_link) {
    color:[.8 .2 .2 .5], type:mesh mesh:'meshes/ur5/collision/upper_arm.stl'  contact:-2 }
body forearm_link { mass:2.275 inertiaTensor:[0.049443313556 0.0 0.0 0.049443313556 0.0 0.004095] }
shape visual forearm_link_1 (forearm_link) {
   type:mesh mesh:'meshes/ur5/visual/Forearm.dae' color:[0.7 0.7 0.7 1.0] colorName:LightGrey  visual }
shape collision forearm_link_0 (forearm_link) {
    color:[.8 .2 .2 .5], type:mesh mesh:'meshes/ur5/collision/forearm.stl'  contact:-2 }
body wrist_1_link { mass:1.219 inertiaTensor:[0.111172755531 0.0 0.0 0.111172755531 0.0 0.21942] }
shape visual wrist_1_link_1 (wrist_1_link) {
   type:mesh mesh:'meshes/ur5/visual/Wrist1.dae' color:[0.7 0.7 0.7 1.0] colorName:LightGrey  visual }
shape collision wrist_1_link_0 (wrist_1_link) {
    color:[.8 .2 .2 .5], type:mesh mesh:'meshes/ur5/collision/wrist_1.stl'  contact:-2 }
body wrist_2_link { mass:1.219 inertiaTensor:[0.111172755531 0.0 0.0 0.111172755531 0.0 0.21942] }
shape visual wrist_2_link_1 (wrist_2_link) {
   type:mesh mesh:'meshes/ur5/visual/Wrist2.dae' color:[0.7 0.7 0.7 1.0] colorName:LightGrey  visual }
shape collision wrist_2_link_0 (wrist_2_link) {
    color:[.8 .2 .2 .5], type:mesh mesh:'meshes/ur5/collision/wrist_2.stl'  contact:-2 }
body wrist_3_link { mass:0.1879 inertiaTensor:[0.0171364731454 0.0 0.0 0.0171364731454 0.0 0.033822] }
shape visual wrist_3_link_1 (wrist_3_link) {
   type:mesh mesh:'meshes/ur5/visual/Wrist3.dae' color:[0.7 0.7 0.7 1.0] colorName:LightGrey  visual }
shape collision wrist_3_link_0 (wrist_3_link) {
    color:[.8 .2 .2 .5], type:mesh mesh:'meshes/ur5/collision/wrist_3.stl'  contact:-2 }
body ee_link { }
shape collision ee_link_0 (ee_link) {
    color:[.8 .2 .2 .5], rel:<T t(-0.01 0 0)> type:box size:[0.01 0.01 0.01 0]  contact:-2 }
body world { }
joint shoulder_pan_joint (base_link shoulder_link) {
   type:hingeX axis:[0 0 1] A:<T t(0.0 0.0 0.089159) E(0.0 0.0 0.0)> limits:[-6.2831853 6.2831853] ctrl_limits:[3.15 150.0 1] }
joint shoulder_lift_joint (shoulder_link upper_arm_link) {
   type:hingeX axis:[0 1 0] A:<T t(0.0 0.13585 0.0) E(0.0 1.570796325 0.0)> limits:[-6.2831853 6.2831853] ctrl_limits:[3.15 150.0 1] }
joint elbow_joint (upper_arm_link forearm_link) {
   type:hingeX axis:[0 1 0] A:<T t(0.0 -0.1197 0.425) E(0.0 0.0 0.0)> limits:[-6.2831853 6.2831853] ctrl_limits:[3.15 150.0 1] }
joint wrist_1_joint (forearm_link wrist_1_link) {
   type:hingeX axis:[0 1 0] A:<T t(0.0 0.0 0.39225) E(0.0 1.570796325 0.0)> limits:[-6.2831853 6.2831853] ctrl_limits:[3.2 28.0 1] }
joint wrist_2_joint (wrist_1_link wrist_2_link) {
   type:hingeX axis:[0 0 1] A:<T t(0.0 0.093 0.0) E(0.0 0.0 0.0)> limits:[-6.2831853 6.2831853] ctrl_limits:[3.2 28.0 1] }
joint wrist_3_joint (wrist_2_link wrist_3_link) {
   type:hingeX axis:[0 1 0] A:<T t(0.0 0.0 0.09465) E(0.0 0.0 0.0)> limits:[-6.2831853 6.2831853] ctrl_limits:[3.2 28.0 1] }
joint ee_fixed_joint (wrist_3_link ee_link) {
   type:rigid A:<T t(0.0 0.0823 0.0) E(0.0 0.0 1.570796325)> }
joint world_joint (world base_link) {
   type:rigid A:<T t(0.0 0.0 0.0) E(0.0 0.0 0.0)> }

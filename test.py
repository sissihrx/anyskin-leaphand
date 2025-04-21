import pybullet as p
import pybullet_data
import time
import random
import numpy as np

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=0, 
    cameraPitch=90,
    cameraTargetPosition=[0, 0, 0.5]
)


# robot_id = p.loadURDF("leap_left_urdf/robot.urdf", useFixedBase=True, flags = (p.URDF_USE_SELF_COLLISION))
robot_id = p.loadURDF("leap_left_urdf/robot.urdf", useFixedBase=True)
p.setCollisionFilterPair(robot_id, robot_id, -1, 13, enableCollision=0)
p.setCollisionFilterPair(robot_id, robot_id, 0, 2, enableCollision=0)
p.setCollisionFilterPair(robot_id, robot_id, 4, 6, enableCollision=0)
p.setCollisionFilterPair(robot_id, robot_id, 8, 10, enableCollision=0)
aabb_min, aabb_max = p.getAABB(robot_id, linkIndex=-1)
z = aabb_max[2]
plane = p.loadURDF("plane.urdf", basePosition=[0, 0, z + 0.017], useFixedBase = True)

positions = np.loadtxt("positionsnew.txt")
data = []
for i in range(len(positions)):
    ok = True
    for j in range(16):
        joint_info = p.getJointInfo(robot_id, j)
        a = positions[i][j] * (joint_info[9] - joint_info[8]) + joint_info[8]
        p.setJointMotorControl2(bodyIndex=robot_id, jointIndex=j, controlMode=p.POSITION_CONTROL, targetPosition=a, force=100)
        if j == 12 and a < -1.8: ok = False
    for _ in range(240): 
        time.sleep(1/2000)
        p.stepSimulation()

    self_collisions = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    self_collisions1 = p.getContactPoints(bodyA=robot_id, bodyB=plane)
    if self_collisions or self_collisions1: ok = False
    if ok == True:
        data.append(positions[i])
    else:
        print(i)

    # if i % 50 == 0:
    #     np.savetxt("positionsnew.txt", np.array(data))

p.disconnect()
# np.savetxt("positionsnew.txt", np.array(data))

#low limit for 12 is-1.8
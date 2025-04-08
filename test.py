import pybullet as p
import pybullet_data
import time
import random
import numpy as np

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

robot_id = p.loadURDF("leap_left_urdf/robot.urdf", useFixedBase=True, flags = (p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS))
p.setCollisionFilterPair(robot_id, robot_id, -1, 3, enableCollision=1)
p.setCollisionFilterPair(robot_id, robot_id, -1, 7, enableCollision=1)
p.setCollisionFilterPair(robot_id, robot_id, -1, 11, enableCollision=1)
p.setCollisionFilterPair(robot_id, robot_id, -1, 15, enableCollision=1)
p.setCollisionFilterPair(robot_id, robot_id, -1, 14, enableCollision=1)
p.setCollisionFilterPair(robot_id, robot_id, -1, 13, enableCollision=1)

positions = np.loadtxt("positions.txt")
data = []
for i in range(len(positions)):
    for j in range(16):
        joint_info = p.getJointInfo(robot_id, j)
        a = positions[i][j] * (joint_info[9] - joint_info[8]) + joint_info[8]
        p.setJointMotorControl2(bodyIndex=robot_id, jointIndex=j, controlMode=p.POSITION_CONTROL, targetPosition=a, force=1000)
    for _ in range(240): 
        p.stepSimulation()

    self_collisions = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    if self_collisions:
        print(i)
        # for c in self_collisions:
        #     pos = c[5]
        #     end = [pos[0], pos[1], pos[2] + 0.2]  
        #     line_id = p.addUserDebugLine(pos, end, [1, 0, 0], lineWidth=5, lifeTime=1)
        #     print(f" Link {c[3]} (vs) Link {c[4]}")
    else:
        data.append(positions[i])

p.disconnect()
np.savetxt("positions.txt", data)

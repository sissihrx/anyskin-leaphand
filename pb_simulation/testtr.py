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

robot_id = p.loadURDF("leap_left_urdf/robot.urdf", useFixedBase=True, flags = (p.URDF_USE_SELF_COLLISION))
p.setCollisionFilterPair(robot_id, robot_id, -1, 13, enableCollision=0)
p.setCollisionFilterPair(robot_id, robot_id, 0, 2, enableCollision=0)
p.setCollisionFilterPair(robot_id, robot_id, 4, 6, enableCollision=0)
p.setCollisionFilterPair(robot_id, robot_id, 8, 10, enableCollision=0)
aabb_min, aabb_max = p.getAABB(robot_id, linkIndex=-1)
z = aabb_max[2]
plane = p.loadURDF("plane.urdf", basePosition=[0, 0, z + 0.009], useFixedBase = True)

positions = np.loadtxt("positionstr.txt")
data = []
data.append(positions[0])
last = positions[0]
for i in range(16): p.setJointMotorControl2(bodyIndex=robot_id, jointIndex=i, controlMode=p.POSITION_CONTROL, targetPosition=positions[0][i], force=100)

for i in range(1, len(positions)):
    # if i % 50 == 0: np.savetxt("positionstr1.txt", np.array(data))
    for k in range(1, 31):
        curr = []
        for j in range(16):
            joint_info = p.getJointInfo(robot_id, j)
            curr.append(positions[i][j])
            a = positions[i][j] * (joint_info[9] - joint_info[8]) + joint_info[8]
            prev = last[j] * (joint_info[9] - joint_info[8]) + joint_info[8]
            a = prev + (a - prev) * k / 30
            p.setJointMotorControl2(bodyIndex=robot_id, jointIndex=j, controlMode=p.POSITION_CONTROL, targetPosition=a, force=100)
        for _ in range(240): 
            # time.sleep(1/2000)
            p.stepSimulation()

        self_collisions = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
        # if self_collisions or k == 20:
        #     if k == 1: break
        #     for x in range(16): p.setJointMotorControl2(bodyIndex=robot_id, jointIndex=j, controlMode=p.POSITION_CONTROL, targetPosition=last[x], force=100)
        #     data.append(last)
        #     break
        # print(i)
        if self_collisions:            
            for x in range(16): p.setJointMotorControl2(bodyIndex=robot_id, jointIndex=j, controlMode=p.POSITION_CONTROL, targetPosition=last[x], force=100)
            print(i)
            break
        elif k == 30:
            data.append(curr)
            last = curr.copy() 
            break


p.disconnect()
# np.savetxt("positionstr1.txt", np.array(data))
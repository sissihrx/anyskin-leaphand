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
plane = p.loadURDF("plane.urdf", basePosition=[0, 0, z + 0.017], useFixedBase = True)

positions = []
for i in range(2000000):
    pos = []
    print(i)
    for j in range(16):
        joint_info = p.getJointInfo(robot_id, j)
        a = random.uniform(joint_info[8], joint_info[9])
        p.setJointMotorControl2(bodyIndex=robot_id, jointIndex=j, controlMode=p.POSITION_CONTROL, targetPosition=a, force=1000)
    for _ in range(240): 
        p.stepSimulation()
    
    good = True
    for j in range(16):
        joint_state = p.getJointState(robot_id, j)
        joint_info = p.getJointInfo(robot_id, j)
        angle = (joint_state[0] - joint_info[8]) / (joint_info[9] - joint_info[8]) # scale 0 and 1
        pos.append(angle)

    self_collisions = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    self_collisions1 = p.getContactPoints(bodyA=robot_id, bodyB=plane)
    if self_collisions or self_collisions1:
        good = False
        # for c in self_collisions:
        #     pos = c[5]
        #     end = [pos[0], pos[1], pos[2] + 0.2]  
        #     line_id = p.addUserDebugLine(pos, end, [1, 0, 0], lineWidth=5, lifeTime=1)
        #     print(f" Link {c[3]} (vs) Link {c[4]}")
    else:
        print(pos)
        positions.append(pos)
    
    if i % 1000 == 0:
        data = np.array(positions)
        np.savetxt("positions1.txt", data)

p.disconnect()
positions = np.array(positions)
np.savetxt("positions1.txt", positions)

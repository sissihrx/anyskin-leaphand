import pybullet as p
import pybullet_data
import time
import random
import numpy as np

p.connect(p.GUI)
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

positions = np.loadtxt("positionsnew.txt")
data = []
last = positions[0]
for i in range(1, 200):
    for k in range(1, 21):
        curr = []
        for j in range(16):
            joint_info = p.getJointInfo(robot_id, j)
            a = positions[i][j] * (joint_info[9] - joint_info[8]) + joint_info[8]
            prev = positions[i-1][j] * (joint_info[9] - joint_info[8]) + joint_info[8]
            a = prev + (a - prev) * k / 20
            curr.append(a)
            p.setJointMotorControl2(bodyIndex=robot_id, jointIndex=j, controlMode=p.POSITION_CONTROL, targetPosition=a, force=100)
        for _ in range(240): 
            # time.sleep(1/2000)
            p.stepSimulation()

        self_collisions = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
        if self_collisions or k == 20:
            if k == 1: break
            for x in range(16): p.setJointMotorControl2(bodyIndex=robot_id, jointIndex=j, controlMode=p.POSITION_CONTROL, targetPosition=last[x], force=100)
            data.append(last)
            break

            for c in self_collisions:
                pos = c[5]
                end = [pos[0], pos[1], pos[2] + 0.2]  
                line_id = p.addUserDebugLine(pos, end, [1, 0, 0], lineWidth=5, lifeTime=1)
                print(f" Link {c[3]} (vs) Link {c[4]}")
        else:
            last = curr.copy()


p.disconnect()
np.savetxt("positionstr.txt", data)


# make 14 shorter in urdf !!
#       make fingertips and middle joints wider (theyre scraping by), thumb tip replace with one big rect box
# make another body thats just the base, turn on collisions between links and that body (prolly dont need)

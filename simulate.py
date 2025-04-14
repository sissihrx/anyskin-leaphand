import pybullet as p
import pybullet_data
import time
import random
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

robot_id = p.loadURDF("leap_left_urdf/robot.urdf", useFixedBase=True, flags = (p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS))
p.setCollisionFilterPair(robot_id, robot_id, -1, 3, enableCollision=1)
p.setCollisionFilterPair(robot_id, robot_id, -1, 7, enableCollision=1)
p.setCollisionFilterPair(robot_id, robot_id, -1, 11, enableCollision=1)
p.setCollisionFilterPair(robot_id, robot_id, -1, 15, enableCollision=1)
# robot_id = p.loadURDF("leap_left_urdf/robot.urdf", useFixedBase=True)


# Print links info
for i in range(16):
    joint_info = p.getJointInfo(robot_id, i)
    joint_name = joint_info[1].decode('utf-8')
    child_link_name = joint_info[12].decode('utf-8')
    child_link_index = i
    parent_link_index = joint_info[16]

    print(f"Joint {i}: {joint_name}")
    print(f"  Parent link: {parent_link_index}")
    print(f"  Child link: {child_link_index}")
    print(f"  Child name: {child_link_name}")

positions = []
for i in range(1000000):
    pos = []
    print(i)
    for j in range(16):
        joint_info = p.getJointInfo(robot_id, j)
        if j == 3: 
            if i % 2 == 0: a = 0.85
            else: a = joint_info[9]
        else: a = 0
        p.setJointMotorControl2(bodyIndex=robot_id, jointIndex=j, controlMode=p.POSITION_CONTROL, targetPosition=a, force=1000)
    for _ in range(240): 
        p.stepSimulation()
        time.sleep(1/1000)
    
    good = True
    for j in range(16):
        joint_state = p.getJointState(robot_id, j)
        joint_info = p.getJointInfo(robot_id, j)
        angle = (joint_state[0] - joint_info[8]) / (joint_info[9] - joint_info[8]) # scale 0 and 1
        pos.append(angle)

    self_collisions = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    if self_collisions:
        good = False
        # for c in self_collisions:
        #     pos = c[5]
        #     end = [pos[0], pos[1], pos[2] + 0.2]  
        #     line_id = p.addUserDebugLine(pos, end, [1, 0, 0], lineWidth=5, lifeTime=1)
        #     print(f" Link {c[3]} (vs) Link {c[4]}")
    else:
        print(pos)
        positions.append(pos)
        for j in range(16):
            if pos[j] < 0 or pos[j] > 1: 
                good = False
                print("not good")
    
    # if i % 50 == 0:
    #     data = np.array(positions)
    #     np.savetxt("positions.txt", data)

p.disconnect()
# positions = np.array(positions)
# np.savetxt("positions.txt", positions)

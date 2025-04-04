import pybullet as p
import pybullet_data
import time
import random

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

robot_id = p.loadURDF("leap_left_urdf/robot.urdf", useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
num_joints = p.getNumJoints(robot_id)

for i in range(100):
    for j in range(16):
        joint_info = p.getJointInfo(robot_id, j)
        if joint_info[2] == p.JOINT_REVOLUTE:
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            random_angle = random.uniform(lower_limit, upper_limit)
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, targetPosition=random_angle)

    for _ in range(1000): 
        p.stepSimulation()
        time.sleep(1 / 1000)

    self_collisions = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    if self_collisions:
        print("Self-collision detected:")
        for c in self_collisions:
            print(f" Link {c[3]} (vs) Link {c[4]}")

input("Press Enter to exit...")
p.disconnect()

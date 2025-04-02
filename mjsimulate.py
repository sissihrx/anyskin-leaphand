import mujoco
import time
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("/Users/sissi/Downloads/leap_left_urdf/mjmodel.xml")
data = mujoco.MjData(model)

print("Joint ranges:")
for i in range(model.njnt):
    print(f"Joint {i}: {model.jnt_range[i]}")

for i in range(model.ngeom):
    model.geom_contype[i] = 1 
    model.geom_conaffinity[i] = 1 

def generatepos(model):
    pos = np.zeros(model.nq)
    # for i in range(model.njnt):
    #     joint_range = model.jnt_range[i]
    #     pos[i] = np.random.uniform(joint_range[0], joint_range[1])
    return pos

model.opt.gravity[:] = 0
data.qpos[:] = np.zeros(16)
data.qvel[:] = np.zeros(16) 
mujoco.mj_step(model, data) 


with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(100):
        pos = generatepos(model)
        last = data.qpos
        print(pos)
        
        b = True
        for j in range(50):
            data.qpos[:] = last + (pos - last) * j / 50
            Kp = 0.5
            Kd = 0.1
            data.qfrc_applied[:] = Kp * (pos - data.qpos) - Kd * data.qvel
            if data.ncon > 0:
                # b = False
                # data.qpos[:] = last
                print(1)
                # break
            mujoco.mj_forward(model, data)
            time.sleep(0.5) 
            viewer.sync()
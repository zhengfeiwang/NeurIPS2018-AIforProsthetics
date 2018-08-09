OBSERVATION_DIM = 85


def process_observation(observation):
    # custom observation space 44 + 3 + 17 + 17 + 4 = 85D
    res = []

    BODY_PARTS = ['femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head']
    JOINTS = ['ground_pelvis', 'hip_r', 'knee_r', 'ankle_r', 'hip_l', 'knee_l', 'ankle_l', 'back']
        
    # body parts positions relative to pelvis - (3 + 1) + (3 + 1) * 10 = 44D
    # pelvis relative position
    res += [0.0, 0.0, 0.0]
    res += [observation["body_pos"]["pelvis"][1]]   # absolute pelvis.y
    pelvis_pos = observation["body_pos"]["pelvis"]
    for body_part in BODY_PARTS:
        # x, y, z - axis
        for axis in range(3):
            res += [observation["body_pos"][body_part][axis] - pelvis_pos[axis]]
        res += [observation["body_pos"][body_part][1]] # absolute height

    # pelvis velocity - 3D
    pelvis_vel = observation["body_vel"]["pelvis"]
    res += pelvis_vel
        
    # joints absolute angle - 6 + 3 + 1 + 1 + 3 + 1 + 1 + 1 = 17D
    for joint in JOINTS:
        for i in range(len(observation["joint_pos"][joint])):
            res += [observation["joint_pos"][joint][i]]
        
    # joints absolute angular velocity - 6 + 3 + 1 + 1 + 3 + 1 + 1 + 1 = 17D
    for joint in JOINTS:
        for i in range(len(observation["joint_vel"][joint])):
            res += [observation["joint_vel"][joint][i]]
        
    # center of mass position and velocity - 2 + 2 = 4D
    for axis in range(2):
        res += [observation["misc"]["mass_center_pos"][axis] - pelvis_pos[axis]]
    for axis in range(2):
        res += [observation["misc"]["mass_center_vel"][axis] - pelvis_vel[axis]]
            
    return res

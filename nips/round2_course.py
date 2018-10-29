

def checkpoints_0(self):
    state_desc = self.get_state_desc()
    prev_state_desc = self.get_prev_state_desc()
    if not prev_state_desc:
        return 0

    reward = 2

    pelvis = state_desc["body_pos"]["pelvis"][1]
    reward -= max(0, 0.70 - pelvis) * 20

    penalty = 0
    # Small penalty for too much activation (cost of transport)
    penalty += np.sum(np.array(self.osim_model.get_activations()) ** 2) * 0.001
    # Big penalty for not matching the vector on the X,Z projection.
    # No penalty for the vertical axis
    penalty += abs(state_desc["body_vel"]["pelvis"][0] - state_desc["target_vel"][0]) * 2
    penalty += abs(state_desc["body_vel"]["pelvis"][2] - state_desc["target_vel"][2]) * 2

    reward -= penalty

    return reward * 0.5


# 接checkpoints_0
def checkpoints_1(self):
    state_desc = self.get_state_desc()
    prev_state_desc = self.get_prev_state_desc()
    if not prev_state_desc:
        return 0

    reward = 2 + state_desc["body_vel"]["pelvis"][0]

    pelvis = state_desc["body_pos"]["pelvis"][1]
    reward -= max(0, 0.70 - pelvis) * 20

    penalty = 0
    # Small penalty for too much activation (cost of transport)
    penalty += np.sum(np.array(self.osim_model.get_activations()) ** 2) * 0.001
    # Big penalty for not matching the vector on the X,Z projection.
    # No penalty for the vertical axis
    penalty += abs(state_desc["body_vel"]["pelvis"][0] - state_desc["target_vel"][0]) * 2
    penalty += abs(state_desc["body_vel"]["pelvis"][2] - state_desc["target_vel"][2]) * 2

    reward -= penalty

    return reward * 0.5


# 接checkpoints_1
def checkpoints_2(self):
    state_desc = self.get_state_desc()
    prev_state_desc = self.get_prev_state_desc()
    if not prev_state_desc:
        return 0

    reward = 2

    pelvis = state_desc["body_pos"]["pelvis"][1]
    reward -= max(0, 0.70 - pelvis) * 20

    penalty = 0
    # Small penalty for too much activation (cost of transport)
    penalty += np.sum(np.array(self.osim_model.get_activations()) ** 2) * 0.001
    # Big penalty for not matching the vector on the X,Z projection.
    # No penalty for the vertical axis
    penalty += abs(state_desc["body_vel"]["pelvis"][0] - state_desc["target_vel"][0]) * 2
    penalty += abs(state_desc["body_vel"]["pelvis"][2] - state_desc["target_vel"][2]) * 2

    reward -= penalty

    return reward * 0.5


# 接checkpoints_0
def checkpoints_3(self):
    state_desc = self.get_state_desc()
    prev_state_desc = self.get_prev_state_desc()
    if not prev_state_desc:
        return 0

    reward = 3

    pelvis = state_desc["body_pos"]["pelvis"][1]
    reward -= max(0, 0.70 - pelvis) * 20

    penalty = 0
    # Small penalty for too much activation (cost of transport)
    penalty += np.sum(np.array(self.osim_model.get_activations()) ** 2) * 0.001
    # Big penalty for not matching the vector on the X,Z projection.
    # No penalty for the vertical axis
    penalty += abs(state_desc["body_vel"]["pelvis"][0] - state_desc["target_vel"][0]) * 2
    penalty += abs(state_desc["body_vel"]["pelvis"][2] - state_desc["target_vel"][2]) * 2

    reward -= penalty

    return reward * 0.5


# 接checkpoints_3
def checkpoints_4(self):
    state_desc = self.get_state_desc()
    prev_state_desc = self.get_prev_state_desc()
    if not prev_state_desc:
        return 0

    reward = 2

    pelvis = state_desc["body_pos"]["pelvis"][1]
    reward -= max(0, 0.70 - pelvis) * 20

    penalty = 0
    # Small penalty for too much activation (cost of transport)
    penalty += np.sum(np.array(self.osim_model.get_activations()) ** 2) * 0.001
    # Big penalty for not matching the vector on the X,Z projection.
    # No penalty for the vertical axis
    penalty += abs(state_desc["body_vel"]["pelvis"][0] - state_desc["target_vel"][0]) * 2
    penalty += abs(state_desc["body_vel"]["pelvis"][2] - state_desc["target_vel"][2]) * 2

    reward -= penalty

    return reward * 0.5



def course_0(self):
    state_desc = self.get_state_desc()
    prev_state_desc = self.get_prev_state_desc()
    if not prev_state_desc:
        return 0

    reward = min(3.0, state_desc["body_vel"]["pelvis"][0]) * 4 + 2

    lean_back = max(0, state_desc["body_pos"]["pelvis"][0] - state_desc["body_pos"]["head"][0] - 0.0)
    reward -= lean_back * 40

    pelvis = state_desc["body_pos"]["pelvis"][1]
    reward -= max(0, 0.70 - pelvis) * 100

    return reward * 0.05


def course_1(self):
    state_desc = self.get_state_desc()
    prev_state_desc = self.get_prev_state_desc()
    if not prev_state_desc:
        return 0

    reward = min(3.0, state_desc["body_vel"]["pelvis"][0]) * 2 + 2 \
             + state_desc["body_vel"]["pros_foot_r"][0] + state_desc["body_vel"]["toes_l"][0]

    lean_back = max(0, state_desc["body_pos"]["pelvis"][0] - state_desc["body_pos"]["head"][0] - 0.2)
    reward -= lean_back * 40

    pelvis = state_desc["body_pos"]["pelvis"][1]
    reward -= max(0, 0.70 - pelvis) * 100

    return reward * 0.05


def course_2_3(self):
    state_desc = self.get_state_desc()
    prev_state_desc = self.get_prev_state_desc()
    if not prev_state_desc:
        return 0

    reward = min(3.0, state_desc["body_vel"]["pelvis"][0]) * 2 + 2 \
             + state_desc["body_vel"]["pros_foot_r"][0] + state_desc["body_vel"]["toes_l"][0]

    front_foot = state_desc["body_pos"]["pros_foot_r"][0]
    back_foot = state_desc["body_pos"]["toes_l"][0]
    dist = max(0.0, front_foot - back_foot - 0.9)
    reward -= dist * 40

    lean_back = max(0, state_desc["body_pos"]["pelvis"][0] - state_desc["body_pos"]["head"][0] - 0.2)
    reward -= lean_back * 40

    pelvis = state_desc["body_pos"]["pelvis"][1]
    reward -= max(0, 0.70 - pelvis) * 100

    return reward * 0.05


def course_4(self):
    state_desc = self.get_state_desc()
    prev_state_desc = self.get_prev_state_desc()
    if not prev_state_desc:
        return 0

    super_reward = 9.0 - (state_desc["body_vel"]["pelvis"][0] - 3.0) ** 2

    reward = super_reward * 0.5 + 1 + state_desc["body_vel"]["pros_foot_r"][0] + state_desc["body_vel"]["toes_l"][0]

    front_foot = state_desc["body_pos"]["pros_foot_r"][0]
    back_foot = state_desc["body_pos"]["toes_l"][0]
    dist = max(0.0, front_foot - back_foot - 0.9)
    reward -= dist * 40

    lean_back = max(0, state_desc["body_pos"]["pelvis"][0] - state_desc["body_pos"]["head"][0] - 0.2)
    reward -= lean_back * 40

    pelvis = state_desc["body_pos"]["pelvis"][1]
    reward -= max(0, 0.70 - pelvis) * 100

    return reward * 0.05


def course_5(self):
    state_desc = self.get_state_desc()
    prev_state_desc = self.get_prev_state_desc()
    if not prev_state_desc:
        return 0

    pelvis_vx = state_desc["body_vel"]["pelvis"][0]
    if pelvis_vx < 1.0:
        reward = -1
    else:
        reward = 9.0 - (pelvis_vx - 3.0) ** 2

    front_foot = state_desc["body_pos"]["pros_foot_r"][0]
    back_foot = state_desc["body_pos"]["toes_l"][0]
    dist = max(0.0, front_foot - back_foot - 0.9)
    reward -= dist * 40

    lean_back = max(0, state_desc["body_pos"]["pelvis"][0] - state_desc["body_pos"]["head"][0] - 0.2)
    reward -= lean_back * 40

    pelvis = state_desc["body_pos"]["pelvis"][1]
    reward -= max(0, 0.7 - pelvis) * 100

    pelvis_z = abs(state_desc["body_pos"]["pelvis"][2])
    reward -= max(0, pelvis_z - 0.6) * 100

    return reward * 0.05

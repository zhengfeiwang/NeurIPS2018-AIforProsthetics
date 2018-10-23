def course1(self):
    state_desc = self.get_state_desc()
    prev_state_desc = self.get_prev_state_desc()
    if not prev_state_desc:
        return 0

    pelvis_vx = state_desc["body_vel"]["pelvis"][0]
    reward = min(1.0, pelvis_vx) * 2 + 1

    lean_back = max(0, state_desc["body_pos"]["pelvis"][0] - state_desc["body_pos"]["head"][0] - 0.2)
    reward -= lean_back * 40

    low_pelvis = max(0, 0.70 - state_desc["body_pos"]["pelvis"][1])
    reward -= low_pelvis * 40

    return reward * 0.05

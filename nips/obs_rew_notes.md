# Round 2 Training Notes - obs & rew

### Course 1
##### observation
- -5: target_vx
- -2: target_vz
- -3: target_vx diff
- -1: target_vz diff
##### reward
- positive: `reward = 2 - (1.25 - state_desc["body_vel"]["pelvis"][0]) ** 2`
- low pelvis penalty: 0.7, 20
- activation penalty
- velocity matching penalty with 2x abs()
- scale = 0.5
---
### Course 2
##### observation
- based on course1
- -4: target_vz
- -6: target_vz diff
##### reward
- positive: 2
- low pelvis penalty: 0.7, 20
- activation penalty
- velocity_x matching penalty with 2x abs()
- velocity_z matching penalty with 3x abs()
- scale = 0.5
---
### Course 3 (Always Failure)
##### observation
- based on course2
- -7: target_vx
- -8, -9: target_vx diff
##### reward
- positive: 2
- low pelvis penalty: 0.7, 20
- activation penalty
- velocity_x matching penalty with 3x abs()
- velocity_z matching penalty with 3x abs()
- scale = 0.5

from osim.env import ProstheticsEnv


class CustomEnv(ProstheticsEnv):

    def __init__(self, visualize=True, integrator_accuracy=5e-5):
        super().__init__(visualize, integrator_accuracy)

    def reward(self):
        reward = super().reward()
        return reward * 0.1


def make_env():
    env = CustomEnv()
    return env

import argparse
import opensim as osim
from osim.http.client import Client
from osim.env import ProstheticsEnv

from ddpg import DDPG

env = ProstheticsEnv(visualize=False)
env.change_model(model='3D', prosthetic=True, difficulty=2, seed=None)

def obs_process(state_desc):
    # Augmented environment from the L2R challenge
    res = []
    pelvis = None

    for body_part in ["pelvis", "head","torso","toes_l","toes_r","talus_l","talus_r"]:
        if body_part in ["toes_r","talus_r"]:
            res += [0] * 9
            continue
        cur = []
        cur += state_desc["body_pos"][body_part][0:2]
        cur += state_desc["body_vel"][body_part][0:2]
        cur += state_desc["body_acc"][body_part][0:2]
        cur += state_desc["body_pos_rot"][body_part][2:]
        cur += state_desc["body_vel_rot"][body_part][2:]
        cur += state_desc["body_acc_rot"][body_part][2:]
        if body_part == "pelvis":
            pelvis = cur
            res += cur[1:]
        else:
            cur_upd = cur
            cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
            cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6,7)]
            res += cur

    for joint in ["ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
        res += state_desc["joint_pos"][joint]
        res += state_desc["joint_vel"][joint]
        res += state_desc["joint_acc"][joint]

    for muscle in sorted(state_desc["muscles"].keys()):
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [state_desc["muscles"][muscle]["fiber_velocity"]]

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
    res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

    return res
    

parser = argparse.ArgumentParser(description='DDPG implemented by PyTorch')
parser.add_argument('--discount', default=0.99, type=float, help='bellman discount')
parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
parser.add_argument('--memory_size', default=1000000, type=int, help='memory size')
parser.add_argument('--hidden1', default=400, type=int, help='number of first fully connected layer')
parser.add_argument('--hidden2', default=300, type=int, help='number of second fully connected layer')
parser.add_argument('--actor_lr', default=1e-4, type=float, help='actor learning rate')
parser.add_argument('--critic_lr', default=1e-3, type=float, help='critic learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
parser.add_argument('--cuda', dest='cuda', action='store_true')
args = parser.parse_args()

nb_states = 158 #env.observation_space.shape[0]
nb_actions = 19 #env.action_space.shape[0]

agent = DDPG(nb_states, nb_actions, args)
agent.load_model('output', 10)

# Settings
remote_base = "http://grader.crowdai.org:1729"
crowdai_token = "0075359aafa067eeeab85a8654798590"

client = Client(remote_base)

# Create environment
observation = client.env_create(token=crowdai_token, env_id="ProstheticsEnv")

# IMPLEMENTATION OF YOUR CONTROLLER
my_controller = agent.select_action

while True:
    observation = obs_process(observation)
    action = my_controller(observation).tolist()

    [observation, reward, done, info] = client.env_step(action, True)
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()

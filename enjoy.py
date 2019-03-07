import argparse
import os

import numpy as np
import torch

from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize


# workaround to unpickle olf model files
import sys
sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment (default: PongNoFrameskip-v4)')
parser.add_argument('--model', default='',
                    help='path to saved pytorch model')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
args = parser.parse_args()

args.det = not args.non_det

env = make_vec_envs(args.env_name, args.seed + 1000, 1,
                    None, None, args.add_timestep, device='cpu',
                    allow_early_resets=False)

os.system('rm -rf videos/*')

video_length = 1000
env = VecVideoRecorder(env, 'videos', record_video_trigger=lambda x: True, video_length=video_length)

# Get a render function
render_func = get_render_func(env)

print('render_func', render_func)
#import ipdb; ipdb.set_trace()

mode = 'human'
#mode = 'rgb_array'

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = torch.load(args.model)
print(actor_critic)

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

if render_func is not None:
    render_func(mode)

obs = env.reset()

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

# while True:
for i in range(video_length):

    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)

    masks.fill_(0.0 if done else 1.0)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    if render_func is not None:
        render_func(mode)

    if done:
        print('end of episode')
        break

print('recorded_frames', env.recorded_frames)

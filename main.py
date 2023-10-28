import csv
from os.path import exists
from pathlib import Path
from bot_env import BotEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
import pandas as pd
import datetime

def make_env(rank, env_conf, df, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = BotEnv(env_conf, df)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    with open('debug.csv', 'w') as ifile:
        writer = csv.writer(ifile, lineterminator='\n')
        writer.writerow(['instance', 'agent?', 'maxstep', 'portfolio_value', 'reward'])

    df = pd.read_csv('AUDUSD5.csv', names=list('dtohlcv'))[list('ohlcv')]
    ep_length = 10000
    sess_path = Path(f'session_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}')

    env_config = {
                'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 
                'init_state': '../has_pokedex_nballs.state', 
                'max_steps': ep_length, 
                'print_rewards': False, 
                'save_frame': False,
                'session_path': sess_path,
                'debug': False, 
                
            }
    
    num_cpu = 3 #64 #46  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config, df) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path, name_prefix='poke')
    #env_checker.check_env(env)
    learn_steps = 40
    file_name = 'session_e41c9eff/poke_38207488_steps' #'session_e41c9eff/poke_250871808_steps'
    
    #'session_bfdca25a/poke_42532864_steps' #'session_d3033abb/poke_47579136_steps' #'session_a17cc1f5/poke_33546240_steps' #'session_e4bdca71/poke_8945664_steps' #'session_eb21989e/poke_40255488_steps' #'session_80f70ab4/poke_58982400_steps'
    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO('MlpPolicy', env, verbose=1, n_steps=ep_length, batch_size=512, n_epochs=1, gamma=0.999, 
                    tensorboard_log="./a2c_cartpole_tensorboard/")
    
    for i in range(learn_steps):
        model.learn(total_timesteps=(ep_length)*num_cpu*50, callback=checkpoint_callback)

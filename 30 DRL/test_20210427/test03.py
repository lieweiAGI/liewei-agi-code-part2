import gym_super_mario_bros
from random import random,randrange
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT,RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import time

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env,RIGHT_ONLY)

done = False
env.reset()

step = 0
while not done:
    action = randrange(len(RIGHT_ONLY))
    state,reward,done,info = env.step(action)
    print(state.shape,done,reward,info)
    env.render()
    # time.sleep(0.02)
    step+=1
env.close()
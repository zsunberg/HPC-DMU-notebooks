# on a system without a head, run 
# xvfb-run -s "-screen 0 1400x900x24" python problem_2_gifs.py

import imageio
import sys

from problem_2 import *

# Perform policy rollout and save to gif
def rollout_save_gif(env, policy, gifname):
    o = env.reset()
    frames = []
    for t in range(1000):
        if t%2==0:
            frames.append(env.render(mode = 'rgb_array'))
        a = policy.action(o)
        a_c = (float(a-4)/2.0,)
        o, _, done, _ = env.step(a_c)
        if done:
            break

    imageio.mimsave(gifname, frames, fps=30)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        gifname = 'mountain_car_viz.gif'
    else:
        gifname = sys.argv[2]
    if len(sys.argv) < 2:
        policyname = 'medium.policy'
    else:
        policyname = sys.argv[1]

    policy = FilePolicy(env, policyname)
    rollout_save_gif(env, policy, gifname)
    print("gif saved to {}".format(gifname))

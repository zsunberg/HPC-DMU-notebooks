import gym
import sys
import random
import numpy as np


env = gym.make('MountainCarContinuous-v0')
env.max_speed = 0.065
env.power = 0.00083
env.min_position = -1.3

# fields that I added for discretization
env.n_pos = 500
env.n_vel = 100


class RandomPolicy:
    def action(self, o):
        return random.randint(1,7)

class EnergyPolicy:
    def action(self, o):
        if o[1] > 0:
            return 7
        else:
            return 1

class RandEnergyPolicy:
    def action(self, o):
        if o[1] > 0:
            return random.randint(3,6)
        else:
            return random.randint(2,5)

class LowEnergyPolicy:
    def action(self, o):
        if o[1] > 0:
            return 5
        else:
            return 3

class FilePolicy:
    def __init__(self, env, filename):
        self.env = env
        self.vec = np.loadtxt(filename)

    def action(self, o):
        return self.vec[state_index(self.env,o)]

def state_index(env, o):
    pos = o[0]
    pi = int((pos - env.min_position)/(env.max_position-env.min_position)*env.n_pos)
    vel = o[1]
    vi = int((vel + env.max_speed)/(2.0*env.max_speed)*env.n_vel)
    return 1 + pi + vi*env.n_pos

def state(env, i):
    pi = (i-1) % env.n_pos
    vi = (i-1-pi)/env.n_pos
    pos = float(pi)/env.n_pos*(env.max_position-env.min_position) + env.min_position
    vel = float(vi)/env.n_vel*(2.0*env.max_speed) - env.max_speed
    return np.array([pos, vel])

def step_to_discrete(env, a):
    a_c = (float(a-4)/2.0,)
    o, r, done, info = env.step(a_c)
    env.state = state(env, state_index(env, o))
    return env.state, r, done, info

def eval_sims(env, policy_vec, n_steps, r_seed, n_sims):
    # env.seed(r_seed)
    random.seed(r_seed)
    minstate = 25185
    min_cs = state(env, minstate)
    maxstate = 25238
    max_cs = state(env, maxstate)
    results = np.zeros(n_sims)
    for k in range(n_sims):
        r_sum = 0.0
        env.reset()
        pos = random.random()*(max_cs[0] - min_cs[0]) + min_cs[0]
        env.state = np.array([pos, 0])
        o = env.state
        for i in range(n_steps):
            a = policy_vec[state_index(env, o)]
            a_c = (float(a-4)/2.0,)
            o, r, done, info = env.step(a_c)
            r_sum += r
            if done:
                break
        results[k] = r_sum
    return results

def evaluate(env, policy_vec, n_steps, r_seed, n_sims):
    results = eval_sims(env, policy_vec, n_steps, r_seed, n_sims)
    return np.mean(results)

if __name__ == "__main__":
    # policy = UpPolicy()
    # policy = RandomPolicy()
    policies = [RandomPolicy(), EnergyPolicy(), RandEnergyPolicy(), LowEnergyPolicy()]

    N = 100000

    s_hist = np.zeros((N,1), dtype=int)
    a_hist = np.zeros((N,1), dtype=int)
    r_hist = np.zeros((N,1), dtype=int)
    sp_hist = np.zeros((N,1), dtype=int)

    env.seed(123)
    random.seed(123)

    o = env.state

    i = -1
    while i < N:
        env.reset()
        # o, r, done, info = step_to_discrete(env, 4)
        # print state_index(env, env.state)
        o = (0,0)
        policy = random.choice(policies)
        for _ in range(500):
            # env.render()
            if i >= 0:
                s_hist[i] = state_index(env, o)
            a = policy.action(o)
            assert type(a) == int, "action was not an int"
            assert a <= 7, "action was greater than 7"
            assert a >= 1, "action was less than 1"
            # o, r, done, info = step_to_discrete(env, a)
            a_c = (float(a-4)/2.0,)
            o, r, done, info = env.step(a_c)
            # sys.stdout.write("\r{} | {} |  {}          ".format(a, r, o))
            # sys.stdout.flush()
            if i >= 0:
                a_hist[i] = a
                r_hist[i] = r*1000
                sp_hist[i] = state_index(env, o)
            i+=1
            if done or i >= N:
                break
        print("completed {}".format(i))

    print

    mat = np.concatenate((s_hist, a_hist, r_hist, sp_hist), axis=1)

    print sum(r_hist)

    np.savetxt("medium.csv", mat, fmt="%d", header='"s","a","r","sp"', comments='', delimiter=',')

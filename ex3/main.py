import numpy as np
from env import Gridworld5x5, JacksCarRental, Action
import algorithms
from scipy.stats import poisson


def grid_world5X5(gamma, theta, a):
    env  = Gridworld5x5()

    dic = {0 : 'L', 1 : 'D', 2 : 'R', 3 : 'U'}

    # V = np.random.rand(5,5)
    V = np.zeros((5, 5), dtype=float)
    # V[0][1] = V[0][3] = 0

    policy = (np.ones((5, 5, 4), dtype=float))/4

    # print("initial V:")
    # print(V)

    if a == 'v':
        V, policy = algorithms.value_iteration(env, V, policy, gamma, theta)
    elif a == 'p':
        V, policy = algorithms.policy_iteration(env, V, policy, gamma, theta)

    print("final value function:")
    print(V)
    
    # print("final policy")
    # print(policy)

    viz =[]

    for i in policy:
        w =[]
        for a in i:
            maximum = max(a)
            # print(maximum)
            indeces = [i for i, val in enumerate(a) if val == maximum]
            string = ''
            for v in indeces:
                string = string + dic[v] 
            w.append(string)
        viz.append(w)

    print("policy vizuvalization:")
    print(viz)


    

def jack_car_rental():
    env = JacksCarRental(False)

    probs, rewards = env._open_to_close(1)
    print(probs)

def main():
    a = input("enter p for policy iteration and v for value iteration:")
    grid_world5X5(gamma= 0.9, theta=0.001, a=a)

    # jack_car_rental()

if __name__ == '__main__':
    main()
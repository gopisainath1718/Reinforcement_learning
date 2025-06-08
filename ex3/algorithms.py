import numpy as np


def policy_improvement(env, V, policy):

    for state in env.state_space:
        q_next_states=[]
        for a in range(env.action_space):
            next_state, reward = env.transitions(state, a)
            q_next_states.append(V[next_state])
        optimal_action = max(q_next_states)
        indeces = [i for i, val in enumerate(q_next_states) if val == optimal_action]

        for a in range(env.action_space):
            policy[state[0], state[1], a] = 0 if a not in indeces else (1/len(indeces))
    
    return policy


def policy_evaluation(env, V, policy, gamma, theta):
    
    while True:
        delta = 0
        for state in env.state_space:
            vs = 0
            for a, a_prob in enumerate(policy[state]):
                vs = vs +  a_prob * env.expected_return(V, state, a, gamma)

            delta = max(delta, np.abs(V[state] - vs))
            V[state] = vs
            
        if delta < theta: break

    return V

def policy_iteration(env, V, policy, gamma, theta):

    while True:

        old_policy = np.copy(policy)

        V = policy_evaluation(env, V, policy, gamma, theta)

        policy = policy_improvement(env, V, policy)

        stable = True
        for state in env.state_space:
            for i, a_prob in enumerate(policy[state]):
                if a_prob != old_policy[state[0], state[1], i]:
                    stable = False
        if stable == True: break

    return V, policy


def value_iteration(env, V, policy, gamma, theta):
    
    while True:
        delta = 0

        for state in env.state_space:
            vs = 0
            returns=[]
            for a in range(env.action_space):
                ret = env.expected_return(V, state, a, gamma)
                returns.append(ret)
                vs = max(returns)

            delta = max(delta, np.abs(V[state] - vs))
            V[state] = vs

        if delta < theta: break
    
    policy = policy_improvement(env, V, policy)

    return V, policy
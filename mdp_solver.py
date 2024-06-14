import argparse
import stormpy
import stormpy.core
import stormpy.pars
import stormpy.examples
import numpy as np
from scipy.optimize import linprog

parser = argparse.ArgumentParser(
    prog="MDP solver",
    description="solves MDPs using Shapley algorithm or Hoffman-Karp algorithm",
)

parser.add_argument("filename")
parser.add_argument("default_values")
parser.add_argument("-a", "--algorithm", choices=["sh", "hk"])

def create_model(prism_program_path, parameter_values):
    """
    Helper function to create the model based on the provided input
    """
    orig_program = stormpy.parse_prism_program(prism_program_path)
    program = orig_program.define_constants(
        stormpy.parse_constants_string(
            orig_program.expression_manager, parameter_values
        )
    )
    options = stormpy.BuilderOptions(True, True)
    options.set_build_state_valuations()
    options.set_build_choice_labels()
    model = stormpy.build_sparse_model_with_options(program, options)
    return model

def shapley(
    prism_program_path, parameter_values, gamma=0.85, epsilon=1e-6
):
    """
    Shapley's algorithm 

    Parameters:
        prism_program_path (str): Path to the PRISM program file
        parameter_values: values for variables in model
        gamma (float): Discount factor
        epsilon (float): Tolerance for convergence

    Returns:
        np.ndarray: Value function for all states
    """
    # Parse the PRISM program
    model = create_model(prism_program_path, parameter_values)

    # Extract states and actions
    num_states = model.nr_states
    num_actions1 = max(len(s.actions) for s in model.states)
    num_actions2 = max(len(s.actions) for s in model.states)

    # Extract transitions and rewards
    reward_model_name = list(model.reward_models.keys())[0]
    state_action_rewards = model.reward_models[reward_model_name].state_action_rewards
    model_rewards = {}
    ind = 0
    for state in model.states:
        for action in state.actions:
            model_rewards[(state.id, action.id)] = state_action_rewards[ind]
            ind += 1

    transitions = np.zeros((num_states, num_actions1, num_actions2, num_states))
    rewards = np.zeros((num_states, num_actions1, num_actions2))

    for s, state in enumerate(model.states):
        choices = state.actions
        for i, choice1 in enumerate(choices):
            for j, _ in enumerate(choices):
                for transition in choice1.transitions:
                    target_state = transition.column
                    probability = transition.value()
                    transitions[s, i, j, target_state] += probability
                    rewards[s, i, j] += model_rewards[
                        (state.id, choice1.id)
                    ]  # Assuming single reward structure

    # Initialize the value function
    V = np.zeros(num_states)

    while True:
        V_prev = V.copy()
        Q = np.zeros((num_states, num_actions1, num_actions2))

        # Compute Q-values for all state-action pairs
        for s in range(num_states):
            for a1 in range(num_actions1):
                for a2 in range(num_actions2):
                    Q[s, a1, a2] = rewards[s, a1, a2] + gamma * np.dot(
                        transitions[s, a1, a2], V_prev
                    )

        # Solve the robust optimization for each state
        for s in range(num_states):
            c = -Q[
                s
            ].flatten()  # Objective function coefficients (minimize -Q is the same as maximize Q)
            A_eq = np.ones((1, num_actions1 * num_actions2))
            b_eq = [1]

            bounds = [(0, 1) for _ in range(num_actions1 * num_actions2)]

            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

            V[s] = -res.fun  # Value of the game for state s

        # Check for convergence
        if np.max(np.abs(V - V_prev)) < epsilon:
            break

    return V


def hoffman_karp(prism_program_path, parameter_values):
    """
    Shapley's algorithm. Due to the simplifications to the problem, using this algorithm is essentially the
    same as running value iteration until convergence once for the entire model

    Parameters:
        prism_program_path (str): Path to the PRISM program file
        parameter_values: values for variables in model
        gamma (float): Discount factor
        epsilon (float): Tolerance for convergence

    Returns:
        np.ndarray: Value function for all states
    """

    # Parsing of the model
    model = create_model(prism_program_path, parameter_values)

    # Define the rewards for each state
    properties = stormpy.parse_properties('Rmax=? [F"goal"]')
    result = stormpy.model_checking(model, properties[0], extract_scheduler=True)
    return result


if __name__ == "__main__":
    args = parser.parse_args()
    filename = args.filename
    default_values = args.default_values
    algorithm = args.algorithm

    if algorithm == "sh":
        print(shapley(filename, default_values))
    elif algorithm == "hk":
        print(hoffman_karp(filename, default_values))
    else:
        raise NotImplementedError

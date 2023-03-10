''' 
This python program implements a naive Bayesian Network using the Pomegranate library
Official Documentation: https://pomegranate.readthedocs.io/en/latest/index.html
Github: https://github.com/jmschrei/pomegranate

A previous move is assumed to be sampled from a human node as well a computer node independently.
The naive Bayesian Network (specifically, the V-DAG (next move conditioned on the moves from the previous rounds)) is initialized
such that the a categorical prior distribution is assumed for each action for both the human and computer node such that each action is equally likely.
Based on the priors, the prediction (next move) node is updated using the appropriate Conditional Probability Table (CPT)
'''

# Import required libraries
import numpy as np
from pomegranate import *
import random
'''
Uncomment this section if you would like to fit your Bayesian Network (V-DAG) using previously collected data
'''
data = np.load("data1.npy", allow_pickle=True) ## Load historical data
data = np.concatenate((data[:-1, :], data[1:,1].reshape(-1,1)), axis=1) ## Re-arrange the array such that column 1 contains previous human moves, column 2 contains previous computer moves and column 3 contains the next computer moves
print(data)
'''
Uncomment this section if you would like to fit your Bayesian Network (V-DAG) known priors or any other pmf values you see fit, else comment
it if you plan to fit it using previously collected data. Please note that either this section or the one above needs to be active during anytime!
'''
# Assume human samples from a categorical distribution comprising of 3 outcomes, each of which are equally likely 
human = DiscreteDistribution({'rock': .28813559, 'paper': 0.33898305, 'scissors': 0.37288136})

# Assume computer samples from a categorical distribution comprising of 3 outcomes, each of which are equally likely 
computer = DiscreteDistribution({'rock': 0.37288136, 'paper': 0.27118644, 'scissors': 0.3559322})
#naive bayes for A
p_s_p_A = 5./16
p_r_p_A = 6.16
p_p_p_A = 5./16
p_s_r_A = 8./23
p_r_r_A = 4./23
p_p_r_A = 11./23
p_p_s_A = 4./20
p_r_s_A = 7./20
p_s_s_A = 9./20

#naive bayes for B
p_s_p_B = 8./16
p_r_p_B = 6.16
p_p_p_B = 2./16
p_s_r_B = 8./23
p_r_r_B = 7./23
p_p_r_B = 8./23
p_p_s_B = 6./20
p_r_s_B = 9./20
p_s_s_B = 5./20

# Prediction is dependent on both the human and computer moves. 
prediction = ConditionalProbabilityTable(
        [[ 'rock', 'rock', 'rock', 1./3 ],
         [ 'rock', 'rock', 'paper', 1./2 ],
         [ 'rock', 'rock', 'scissors', 1./6 ],
         [ 'rock', 'paper', 'rock', 0],
         [ 'rock', 'paper', 'paper', .2 ],
         [ 'rock', 'paper', 'scissors', .8 ],
         [ 'rock', 'scissors', 'rock', 1./3 ],
         [ 'rock', 'scissors', 'paper', 1./3 ],
         [ 'rock', 'scissors', 'scissors', 1./3 ],
         [ 'paper', 'rock', 'rock', .5],
         [ 'paper', 'rock', 'paper', 0],
         [ 'paper', 'rock', 'scissors', .5],
         [ 'paper', 'paper', 'rock', 2./3],
         [ 'paper', 'paper', 'paper', 1./6],
         [ 'paper', 'paper', 'scissors', 1./6 ],
         [ 'paper', 'scissors', 'rock', 1./2 ],
         [ 'paper', 'scissors', 'paper', .4 ],
         [ 'paper', 'scissors', 'scissors', .1 ],
         [ 'scissors', 'rock', 'rock', 1./4 ],
         [ 'scissors', 'rock', 'paper', 1./4 ],
         [ 'scissors', 'rock', 'scissors', 1./2 ],
         [ 'scissors', 'paper', 'rock', .8 ],
         [ 'scissors', 'paper', 'paper', 0 ],
         [ 'scissors', 'paper', 'scissors', .2 ],
         [ 'scissors', 'scissors', 'rock', .2],
         [ 'scissors', 'scissors', 'paper', .4 ],
         [ 'scissors', 'scissors', 'scissors', .4]], [human, computer])

# State objects hold both the distribution and the asscoiated node/state name. Both state and node mean the same in regard to the Bayesian Network
s1 = State(human, name="human")
s2 = State(computer, name="computer")
s3 = State(prediction, name="prediction")


# Create the Bayesian network object using a suitable name
model = BayesianNetwork("Rock Paper Scissors")

# Add the three states to the network 
model.add_states(s1, s2, s3)

# Add edges which represent conditional dependencies, where the prediction node is 
# conditionally dependent on its parent nodes (Prediction is dependent on both human and computer moves)
model.add_edge(s1, s3)
model.add_edge(s2, s3)

# Finalize the Bayesian Network
model.bake()

# Uncomment only if you want to directly fit your Bayesian Network using data. 
model.fit(data)

# Prints the model summary (all marginal and conditional probability distributions)
print ("Bayesian Network Summary: {}".format(model))

# Uncomment this line if you would like to predict, in this case the joint probability of (rock, paper) being the previous round moves and the next being scissors
print (model.probability([['rock', 'paper', 'rock']]))

# The following line returns the action that maximizes P(prediction|human_move,computer_move)
prediction = model.predict([['rock', 'paper', None]])
print ("Argmax_Prediction:{}".format(prediction[-1][-1]))

# To generate predictions probabilities for each of the possible actions, provide as evidence "Human": "Rock" and "Computer": "Paper" to your model
predictions = model.predict_proba({"human": "rock", "computer": "paper"})

# Print prediction probabilities for each node
for node, prediction in zip(model.states, predictions):
    if isinstance(prediction, str):
        print(f"{node.name}: {prediction}") ## Prints the current state (previous moves) of the Human and Computer Nodes in your Bayesian Network
    else:
        print(f"{node.name}")
        for value, probability in prediction.parameters[0].items():
            print(f"    {value}: {probability:.4f}")  ## Prints the probability for each possible action given the current state of the parents in your Bayesian Network


def bayes_function_vdag(A_move, B_move):
    # p_r = 0.28813559
    # p_p = 0.33898305
    # p_s = 0.37288136
    if A_move == 'rock' and B_move == 'rock':
        next_move = 'paper'
    elif A_move == 'rock' and B_move == 'paper':
        next_move = 'scissors'
    elif A_move == 'rock' and B_move == 'scissors':
        move = random.randint(1, 3)
        if move == 1:
            next_move = 'rock'
        elif move == 2:
            next_move = 'paper'
        else:
            next_move = 'scissors'
    elif A_move == 'paper' and B_move == 'paper':
        next_move = 'rock'
    elif A_move == 'paper' and B_move == 'rock':
        move = random.randint(1, 2)
        if move == 1:
            next_move = 'rock'
        else:
            next_move = 'scissors'
    elif A_move == 'paper' and B_move == 'scissors':
        next_move = 'rock'
    elif A_move == 'scissors' and B_move == 'scissors':
        move = random.randint(1, 2)
        if move == 1:
            next_move = 'paper'
        else:
            next_move = 'scissors'
    elif A_move == 'scissors' and B_move == 'paper':
        next_move = 'rock'
    else:
        next_move = 'scissors'
    return next_move


def bayes_function_ivdag(A_move, B_move):
    p_s_p_B = 8. / 16
    p_r_p_B = 6./16
    p_p_p_B = 2. / 16
    p_s_r_B = 8. / 23
    p_r_r_B = 7. / 23
    p_p_r_B = 8. / 23
    p_p_s_B = 6. / 20
    p_r_s_B = 9. / 20
    p_s_s_B = 5. / 20
    p_s_p_A = 5. / 16
    p_r_p_A = 6.16
    p_p_p_A = 5. / 16
    p_s_r_A = 8. / 23
    p_r_r_A = 4. / 23
    p_p_r_A = 11. / 23
    p_p_s_A = 4. / 20
    p_r_s_A = 7. / 20
    p_s_s_A = 9. / 20
    p_p = 16./59
    p_r = 23./59
    p_s = 20./59
    if A_move == 'rock' and B_move == 'rock':
        prob1 = p_r * p_r_r_A * p_r_r_B
        prob2 = p_p * p_r_p_A * p_r_p_B
        prob3 = p_s * p_r_s_A * p_r_s_B
        if max(prob1, prob2, prob3) == prob1:
            next_move = 'rock'
        elif max(prob1, prob2, prob3) == prob2:
            next_move = 'paper'
        else:
            next_move = 'scissors'
    elif A_move == 'rock' and B_move == 'paper':
        prob1 = p_r * p_r_r_A * p_p_r_B
        prob2 = p_p * p_r_p_A * p_p_p_B
        prob3 = p_s * p_r_s_A * p_p_s_B
        if max(prob1, prob2, prob3) == prob1:
            next_move = 'rock'
        elif max(prob1, prob2, prob3) == prob2:
            next_move = 'paper'
        else:
            next_move = 'scissors'
    elif A_move == 'rock' and B_move == 'scissors':
        prob1 = p_r * p_r_r_A * p_s_r_B
        prob2 = p_p * p_r_p_A * p_s_p_B
        prob3 = p_s * p_r_s_A * p_s_s_B
        if max(prob1, prob2, prob3) == prob1:
            next_move = 'rock'
        elif max(prob1, prob2, prob3) == prob2:
            next_move = 'paper'
        else:
            next_move = 'scissors'
    elif A_move == 'paper' and B_move == 'paper':
        prob1 = p_r * p_p_r_A * p_p_r_B
        prob2 = p_p * p_p_p_A * p_p_p_B
        prob3 = p_s * p_p_s_A * p_p_s_B
        if max(prob1, prob2, prob3) == prob1:
            next_move = 'rock'
        elif max(prob1, prob2, prob3) == prob2:
            next_move = 'paper'
        else:
            next_move = 'scissors'
    elif A_move == 'paper' and B_move == 'rock':
        prob1 = p_r * p_p_r_A * p_r_r_B
        prob2 = p_p * p_p_p_A * p_r_p_B
        prob3 = p_s * p_p_s_A * p_r_s_B
        if max(prob1, prob2, prob3) == prob1:
            next_move = 'rock'
        elif max(prob1, prob2, prob3) == prob2:
            next_move = 'paper'
        else:
            next_move = 'scissors'
    elif A_move == 'paper' and B_move == 'scissors':
        prob1 = p_r * p_p_r_A * p_s_r_B
        prob2 = p_p * p_p_p_A * p_s_p_B
        prob3 = p_s * p_p_s_A * p_s_s_B
        if max(prob1, prob2, prob3) == prob1:
            next_move = 'rock'
        elif max(prob1, prob2, prob3) == prob2:
            next_move = 'paper'
        else:
            next_move = 'scissors'
    elif A_move == 'scissors' and B_move == 'scissors':
        prob1 = p_r * p_s_r_A * p_s_r_B
        prob2 = p_p * p_s_p_A * p_s_p_B
        prob3 = p_s * p_s_s_A * p_s_s_B
        if max(prob1, prob2, prob3) == prob1:
            next_move = 'rock'
        elif max(prob1, prob2, prob3) == prob2:
            next_move = 'paper'
        else:
            next_move = 'scissors'
    elif A_move == 'scissors' and B_move == 'paper':
        prob1 = p_r * p_s_r_A * p_p_r_B
        prob2 = p_p * p_s_p_A * p_p_p_B
        prob3 = p_s * p_s_s_A * p_p_s_B
        if max(prob1, prob2, prob3) == prob1:
            next_move = 'rock'
        elif max(prob1, prob2, prob3) == prob2:
            next_move = 'paper'
        else:
            next_move = 'scissors'
    else:
        prob1 = p_r * p_s_r_A * p_r_r_B
        prob2 = p_p * p_s_p_A * p_r_p_B
        prob3 = p_s * p_s_s_A * p_r_s_B
        if max(prob1, prob2, prob3) == prob1:
            next_move = 'rock'
        elif max(prob1, prob2, prob3) == prob2:
            next_move = 'paper'
        else:
            next_move = 'scissors'
    return next_move

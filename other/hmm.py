
# Based on CS50 AI
# https://www.youtube.com/watch?v=uQmYZTTqDC0&list=PLnrZOBR0x7yq6_p-DywsuH1R559NU2xLp&index=2

from pomegranate import *
import numpy as np

# Observation model for each state / Emission Probabilities
sun = DiscreteDistribution({
    "umbrella": 0.2,
    "no umbrella": 0.8
})

rain = DiscreteDistribution({
    "umbrella": 0.9,
    "no umbrella": 0.1
})

states = [sun, rain]

# Transition Distribution
transitions = np.array(
    [[0.8, 0.2],
     [0.3, 0.7]]
)

# Starting Distribution
starts = ([0.5, 0.5])


# Create the model
model = HiddenMarkovModel.from_matrix(
    transitions, states, starts,
    state_names=["sun", "rain"]
)
model.bake()


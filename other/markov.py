# Based on CS50 AI
# https://www.youtube.com/watch?v=uQmYZTTqDC0&list=PLnrZOBR0x7yq6_p-DywsuH1R559NU2xLp&index=2

from pomegranate import *


# Starting Distribution
start = DiscreteDistribution({
    "sun": 0.5,
    "rain": 0.5
})


# Transition Distribution
transitions = ConditionalProbabilityTable([
    ["sun", "sun", 0.8],
    ["sun", "rain", 0.2],
    ["rain", "sun", 0.3],
    ["rain", "rain", 0.7]
], [start])

# Markov Chain
model = MarkovChain([start, transitions])

# Sample 50 states
print(model.sample(50))



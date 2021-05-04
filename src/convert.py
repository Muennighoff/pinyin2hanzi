import argparse
import json

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default="./data/input.txt", help="Input Pinyin File.")
    parser.add_argument("--output", type=str, default="./data/output.txt", help="Output Hanzi File.")

    args = parser.parse_args()
    return args

def read_json(path):
    with open(path) as f:
        return json.load(f)

def read_txt(path):
    """
    Creates list of lists observations
    """
    observations = []
    with open(path, 'r') as f:
        for line in f:
            observations.append(line.strip().split(" "))
    return observations

def write_txt(observations, path):
    with open(path, 'w', encoding="utf-8") as f:
        for line in observations:
            f.write(" ".join(line) + "\n")

def viterbi(start, emission, transition, pin2han, line):
    """
    Given a sequence of observations, decode the most likely hidden states
    """

    # Check if all correct pinyin
    for p in line:
        assert p in pin2han.keys(), f"Incorrect Pinyin: {p}"

    # Initialization
    # delta_0(i) = pi_i * b_i(O_0)
    # The start probability for each Hanzi is the product of :
    # a) Its start probability 
    # b) The probability of the observation at t=0 (start) given that hanzi
    tracker = [{}]

    # Get possible hidden states given observation at t=0
    cur_states = pin2han[line[0]]
    # Calculate Probability for each possible state
    for state in cur_states:
        # In case it does not exist in start / emission dict
        start["data"].setdefault(state, start["default"]) 
        emission["data"].setdefault(state, {}) 
        emission["data"][state].setdefault(line[0], emission["default"])

        # Calculate Prob
        score = start["data"][state] * emission["data"][state][line[0]]

        # Keep track of score & path
        tracker[0].setdefault(state, {})
        tracker[0][state].setdefault(state, {})
        tracker[0][state][state] = {"score": score, "path": [state]}

    # tracker is of structure:
    # [cur_state_0: {prev_state_0: {score:, SCORE, path: PATH}, prev_state_1: {...}}, cur_state_1: {...}]
    # [{'亲': {'亲': {'score': 5.603259762138639e-07, 'path': ['亲']}}, '倾': {'倾': {'score': 1.9838110150970737e-05, 'path': ['倾']}}, ..]
    # Eventually it will look like:
    # [{ ... '挖‘：{'举': {'score': 3.6294346118865776e-11, 'path': ['书', '举', '挖']}, '据': {'score': 2.8896701878514984e-07, 'path': ['数', '据', '挖']},... } ... }]
    # i.e. a list of dicts of dicts

    # t > 0
    # delta_t(j) = max over all prev deltas i: delta_t-1(i) * transition_prob_i->j * b_j(O_t)
    # I.e. for each prev delta multiply it by the transition & emission prob; take the max

    for t in range(1, len(line)):

        prev_states = cur_states
        cur_states = pin2han[line[t]]

        # Dynamically delete history, as we keep track of path anyways
        if len(tracker) == 2:
            tracker = [tracker[-1]]
        tracker.append({})

        for state in cur_states:
            emission["data"].setdefault(state, {}) 
            emission["data"][state].setdefault(line[t], emission["default"])

            tracker[1].setdefault(state, {})

            for state_p in prev_states:
                transition["data"].setdefault(state_p, {})
                transition["data"][state_p].setdefault(state, transition["default"])

                tracker[1][state].setdefault(state_p, {})

                # Pick the highest of the prev_states to prev_prev_states combinations
                sort_p = sorted(tracker[0][state_p], key=lambda state_p_p: tracker[0][state_p][state_p_p]["score"], reverse=True)
                state_p_dict = tracker[0][state_p][sort_p[0]]

                # Calculate Prob
                score = state_p_dict["score"] * transition["data"][state_p][state] * emission["data"][state][line[t]]
                # Keep track of score & path
                tracker[1][state][state_p] = {"score": score, "path": state_p_dict["path"] + [state]}             

    top_states = []
    for state in tracker[-1]:
        state_top = sorted(tracker[-1][state], key=lambda state_p: tracker[-1][state][state_p]["score"], reverse=True)
        state_top_dict = tracker[-1][state][state_top[0]]
        top_states.append(state_top_dict)

    result = sorted(top_states, key=lambda state_dict: state_dict["score"], reverse=True)

    print("Prediction", result[0])

    return result[0]["path"]

def main(args):
    start      = read_json("./src/start.json")
    emission   = read_json("./src/emission.json")
    transition = read_json("./src/transition.json")
    pin2han    = read_json("./src/pin2han.json")

    observations = read_txt(args.input)

    for i, line in enumerate(observations):
        observations[i] = viterbi(start, emission, transition, pin2han, line)

    write_txt(observations, args.output)

if __name__ == "__main__":
    args = parse_args()
    main(args)
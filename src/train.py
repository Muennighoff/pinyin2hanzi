import json
import codecs

from pypinyin import lazy_pinyin

def read_hp(emission, path="./src/hanzipinyin.txt"):
    hanzi_set = set()
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            hanzi_set.add(line.split("=")[0])
    return hanzi_set

def read_train(start, emission, transition, hanzi_set, path="./src/2016-11.txt", encoding="utf-8"):
    """
    Use different ecoding than utf-8 to handle mojibake.
    """
    with codecs.open(path, 'r', encoding=encoding, errors='ignore') as f:
        for i, line in enumerate(f):
            json_data = json.loads(line)
            text = json_data["html"]

            text = text.replace(".", "。")
            for sent in text.split("。"):
                # Reduce to hanzi only (Potential Problem: Will reduce 好，但是 to 好但是)
                sent = "".join([c for c in sent if c in hanzi_set])
                
                if len(sent) < 2:
                    continue

                # Add start
                start.setdefault(sent[0], 0)
                start[sent[0]] += 1

                pinyin = lazy_pinyin(sent)

                for j, (c, p) in enumerate(zip(list(sent), pinyin)):
                    # Add Emission
                    emission.setdefault(c, {})
                    emission[c].setdefault(p, 0)

                    emission[c][p] += 1
                    
                    if j == (len(sent)-1):
                        break

                    # Add Transition
                    transition.setdefault(c, {})
                    transition[c].setdefault(sent[j+1], 0)

                    transition[c][sent[j+1]] += 1

def write_json(data, filename):
    with open(filename, 'w') as outfile:
        data = json.dumps(data, indent=4, sort_keys=True)
        outfile.write(data)

# Trains Hidden Markov Model
def main():
    # Start Probabilities - Initial hidden state probabilities
    start      = {}     # {'我':0.006, '了':0.00001}
    # Emission Probabilities - Conditional Probability of Observation given hidden state
    emission   = {}     # {'我': {'wo':1.0}, '了':{'liao':0.5, 'le':0.5}}
    # Transition Probabilities - Conditional Probability of next hidden state given hidden state
    transition = {}     # {'我': {'们':0.3, '是':0.8}, '了': {}}

    # Set of all hanzi
    hanzi_set = read_hp("./src/hanzipinyin.txt")

    # Fill all
    # Time: 200lines ~ 6s > 210 000 ~ 1h30
    read_train(start, emission, transition, hanzi_set, "./src/2016-11.txt", "utf-8")
    #read_train(start, emission, transition, hanzi_set, "./src/2016-02.txt", "gb2312")
    #read_train(start, emission, transition, hanzi_set, "./src/2016-04.txt", "gb2312")
    #read_train(start, emission, transition, hanzi_set, "./src/2016-05.txt", "gb2312")
    #read_train(start, emission, transition, hanzi_set, "./src/2016-06.txt", "gb2312")
    #read_train(start, emission, transition, hanzi_set, "./src/2016-07.txt", "gb2312")
    #read_train(start, emission, transition, hanzi_set, "./src/2016-08.txt", "gb2312")
    #read_train(start, emission, transition, hanzi_set, "./src/2016-09.txt", "gb2312")
    #read_train(start, emission, transition, hanzi_set, "./src/2016-10.txt", "gb2312")

    # Turn into probas
    # Total Start occurences + "1" base for all Hanzi without start occurence
    total = sum(start.values()) + len(hanzi_set - set(start.values()))
    for k,v in start.items():
        start[k] = v/total
    start = {"data": start, "default": 1./total}

    # Also write a pin2han dict at the same time to know possible states given an observation
    pin2han = {}
    for k,v in emission.items():
        total = sum(v.values())
        for v2,s in v.items():
            emission[k][v2] = s/total

            pin2han.setdefault(v2, [])
            if k not in pin2han[v2]:
                pin2han[v2].append(k)
    emission = {"data": emission, "default": 1e-200}

    for k,v in transition.items():
        total = sum(v.values()) + len(hanzi_set - set(v.values())) # Counting each as 1; i.e. every hanzi has 1 count of appearing next
        for v2,s in v.items():
            transition[k][v2] = s/total
    transition = {"data": transition, "default": 1./len(hanzi_set)}

    # Write to JSON
    write_json(start, "./src/start.json")
    write_json(emission, "./src/emission.json")
    write_json(transition, "./src/transition.json")

    write_json(pin2han, "./src/pin2han.json")

if __name__ == "__main__":
    main()
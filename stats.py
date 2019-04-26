import numpy as np
pathA = "./group17.perplexityA"
pathB = "./group17.perplexityB"
pathC = "./group17.perplexityC"

perps = []
with open(pathC) as lines:
    vals = []
    for line in lines:
        a = float(line)
        vals.append(a)
    vals = np.array(vals)
    print("min: {}, max: {}, mean: {}, median: {}".format(np.min(vals), np.max(vals), np.mean(vals), np.median(vals)))

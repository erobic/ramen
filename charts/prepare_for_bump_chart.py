import numpy as np
import csv
raw_format = np.array([['Natural Image Understanding', 64.55, 57.08, 67.39, 54.35, 60.96, 65.96],
                       ['Task Generalization', 68.82, 65.57, 71.1, 66.43, 65.06, 72.52],
                       ['Bias Resistance', 38.01, 38.32, 39.31, 31.96, 26.7, 39.21],
                       ['Concept Compositionality', 57.01, 56.45, 57.36, 50.99, 48.11, 58.92],
                       ['Compositonal Reasoning', 80.04, 46.73, 90.79, 98, 95.97, 96.92]])

flattened_data = [["Method", "Generalization Test", "Accuracy"]]
methods = ['UpDn', 'QCG', 'BAN', 'MAC', 'RN', 'RAMEN']
ability_to_method_scores = {}
for r in raw_format:
    for cix, score in enumerate(r):
        if cix == 0:
            continue
        method = methods[cix  - 1]
        ability = r[0]
        flattened_data.append([method, ability, score])
        if ability not in ability_to_method_scores:
            ability_to_method_scores[ability] = []
        ability_to_method_scores[ability].append([method, ability, score])

# Sort the methods within each ability by scores and then assign ranks
ranked_flattened_data = []
for ability in ability_to_method_scores:
    ability_to_method_scores[ability] = sorted(ability_to_method_scores[ability], key=lambda x: x[2], reverse=True)
    for rank_ix, method_details in enumerate(ability_to_method_scores[ability]):
        method_details.append(len(methods) - rank_ix)
    ranked_flattened_data += ability_to_method_scores[ability]


print("ranked_flattened_data {}".format(np.array(ranked_flattened_data)))

with open('data_for_bump_chart.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter = '|')
    for rfd in ranked_flattened_data:
        writer.writerow(rfd)
#np.savetxt("data_for_bump_chart.csv", np.array(ranked_flattened_data), delimiter=",")
from config.definitions import ROOT_DIR
import numpy as np

predicted_pull = np.load(ROOT_DIR + "/neural_networks/feedforward/predicted_data/predicted_pull.npy")
predicted_push = np.load(ROOT_DIR + "/neural_networks/feedforward/predicted_data/predicted_push.npy")
predicted_shake = np.load(ROOT_DIR + "/neural_networks/feedforward/predicted_data/predicted_shake.npy")
predicted_twist = np.load(ROOT_DIR + "/neural_networks/feedforward/predicted_data/predicted_twist.npy")

pull_struct = {"name": "pull", "TP": 0, "FN": 0, "FP": 0}
push_struct = {"name": "push", "TP": 0, "FN": 0, "FP": 0}
shake_struct = {"name": "shake", "TP": 0, "FN": 0, "FP": 0}
twist_struct = {"name": "twist", "TP": 0, "FN": 0, "FP": 0}

for i in range(0, len(predicted_pull)):

    if predicted_pull[i][-1] == 0.0:
        pull_struct["TP"] += 1
    else:
        pull_struct["FP"] += 1

    if predicted_pull[i][-1] == 1.0:
        push_struct["FN"] += 1

    if predicted_pull[i][-1] == 2.0:
        shake_struct["FN"] += 1

    if predicted_pull[i][-1] == 3.0:
        twist_struct["FN"] += 1


for i in range(0, len(predicted_push)):

    if predicted_push[i][-1] == 1.0:
        push_struct["TP"] += 1
    else:
        push_struct["FP"] += 1

    if predicted_push[i][-1] == 0.0:
        pull_struct["FN"] += 1

    if predicted_push[i][-1] == 2.0:
        shake_struct["FN"] += 1

    if predicted_push[i][-1] == 3.0:
        twist_struct["FN"] += 1


for i in range(0, len(predicted_shake)):

    if predicted_shake[i][-1] == 2.0:
        shake_struct["TP"] += 1
    else:
        shake_struct["FP"] += 1

    if predicted_shake[i][-1] == 0.0:
        pull_struct["FN"] += 1

    if predicted_shake[i][-1] == 1.0:
        push_struct["FN"] += 1

    if predicted_shake[i][-1] == 3.0:
        twist_struct["FN"] += 1

for i in range(0, len(predicted_twist)):

    if predicted_twist[i][-1] == 3.0:
        twist_struct["TP"] += 1
    else:
        twist_struct["FP"] += 1

    if predicted_twist[i][-1] == 0.0:
        pull_struct["FN"] += 1

    if predicted_twist[i][-1] == 1.0:
        push_struct["FN"] += 1

    if predicted_twist[i][-1] == 2.0:
        shake_struct["FN"] += 1


list_structs = [pull_struct, push_struct, shake_struct, twist_struct]

print("--------------------------------------------------------------")
print("--------------------------------------------------------------")

for struct in list_structs:

    print("Interaction: " + struct["name"])

    p = struct["TP"] / (struct["TP"] + struct["FP"])

    print("Precision: " + str(round(p * 100, 2)) + "%")

    r = struct["TP"] / (struct["TP"] + struct["FN"])

    print("Recall: " + str(round(r * 100, 2)) + "%")

    f = 2 * ((p*r)/(p+r))

    print("F1: " + str(round(f * 100, 2)) + "%")

    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")




#!/usr/bin/env python3

from itertools import product
from json import JSONEncoder
import statistics
import matplotlib.pyplot as plt
from keras import Input, Model
import tensorflow as tf
from keras.layers import Dense, LSTM, Layer
import numpy as np
from keras.layers import Lambda
from keras import backend as K


def plot_confusion_matrix_percentage(confusion_matrix, display_labels=None, cmap="viridis",
                                     xticks_rotation="horizontal", title="Confusion Matrix", decimals=.1):
    colorbar = True
    im_kw = None
    fig, ax = plt.subplots()
    cm = confusion_matrix
    n_classes = cm.shape[0]

    default_im_kw = dict(interpolation="nearest", cmap=cmap)
    im_kw = im_kw or {}
    im_kw = {**default_im_kw, **im_kw}

    im_ = ax.imshow(cm, **im_kw)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)

    text_ = np.empty_like(cm, dtype=object)

    # print text with appropriate color depending on background
    thresh = (cm.max() + cm.min()) / 2.0

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        # text_cm = format(cm[i, j], ".1f") + " %"
        text_cm = format(cm[i, j], str(decimals)+"f")
        text_[i, j] = ax.text(
            j, i, text_cm, ha="center", va="center", color=color
        )

    if display_labels is None:
        display_labels = np.arange(n_classes)
    else:
        display_labels = display_labels
    if colorbar:
        fig.colorbar(im_, ax=ax)
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=display_labels,
        yticklabels=display_labels,
        ylabel="True label",
        xlabel="Predicted label",
    )

    ax.set_ylim((n_classes - 0.5, -0.5))
    fig.suptitle(title)
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {iteration} out of {total} Training & Test | ({percent}% {suffix})', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def prediction_classification(cla, true_out, dec_pred, dictionary, pred):
    if true_out == cla and dec_pred == cla:
        dictionary["true_positive"] = np.append(dictionary["true_positive"], pred, axis=0)
    elif true_out != cla and dec_pred == cla:
        dictionary["false_positive"] = np.append(dictionary["false_positive"], pred, axis=0)
    elif true_out == cla and dec_pred != cla:
        dictionary["false_negative"] = np.append(dictionary["false_negative"], pred, axis=0)
    elif true_out != cla and dec_pred != cla:
        dictionary["true_negative"] = np.append(dictionary["true_negative"], pred, axis=0)


def prediction_classification_absolute(cla, true_out, dec_pred, dictionary):
    if true_out == cla and dec_pred == cla:
        dictionary["true_positive"] += 1
    elif true_out != cla and dec_pred == cla:
        dictionary["false_positive"] += 1
    elif true_out == cla and dec_pred != cla:
        dictionary["false_negative"] += 1
    elif true_out != cla and dec_pred != cla:
        dictionary["true_negative"] += 1



def values_contabilization(origin_dict, dest_dict):
    for indice, val in enumerate(origin_dict):
        dest_dict["accumulated"][indice] += val
        dest_dict["number"][indice] += 1


def mean_calc(origin_dict, dest_list, ind):
    mean = origin_dict["accumulated"][ind] / origin_dict["number"][ind]
    dest_list.append(mean)


def group_classification(origin_dict, dest_dict):
    dest_dict["true_positive"].append(len(origin_dict["true_positive"]))
    dest_dict["false_positive"].append(len(origin_dict["false_positive"]))
    dest_dict["false_negative"].append(len(origin_dict["false_negative"]))
    dest_dict["true_negative"].append(len(origin_dict["true_negative"]))


def filling_metrics_table(pull_metrics, push_metrics, shake_metrics, twist_metrics):
    data = [
        ["", "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean F1", ],
        ["PULL", str(round(statistics.mean(pull_metrics["accuracy"]), 4)),
         str(round(statistics.mean(pull_metrics["precision"]), 4)),
         str(round(statistics.mean(pull_metrics["recall"]), 4)), str(round(statistics.mean(pull_metrics["f1"]), 4)), ],
        ["PUSH", str(round(statistics.mean(push_metrics["accuracy"]), 4)),
         str(round(statistics.mean(push_metrics["precision"]), 4)),
         str(round(statistics.mean(push_metrics["recall"]), 4)), str(round(statistics.mean(push_metrics["f1"]), 4)), ],
        ["SHAKE", str(round(statistics.mean(shake_metrics["accuracy"]), 4)),
         str(round(statistics.mean(shake_metrics["precision"]), 4)),
         str(round(statistics.mean(shake_metrics["recall"]), 4)), str(round(statistics.mean(shake_metrics["f1"]), 4)), ],
        ["TWIST", str(round(statistics.mean(twist_metrics["accuracy"]), 4)),
         str(round(statistics.mean(twist_metrics["precision"]), 4)),
         str(round(statistics.mean(twist_metrics["recall"]), 4)), str(round(statistics.mean(twist_metrics["f1"]), 4)), ],
    ]

    return data


def filling_metrics_table_n(pull_metrics, push_metrics, shake_metrics, twist_metrics, n):
    data = [
        ["test " + str(n), "Accuracy", "Precision", "Recall", "F1", ],
        ["PULL", str(round(pull_metrics["accuracy"][n], 4)), str(round(pull_metrics["precision"][n], 4)),
         str(round(pull_metrics["recall"][n], 4)), str(round(pull_metrics["f1"][n], 4)), ],
        ["PUSH", str(round(push_metrics["accuracy"][n], 4)), str(round(push_metrics["precision"][n], 4)),
         str(round(push_metrics["recall"][n], 4)), str(round(push_metrics["f1"][n], 4)), ],
        ["SHAKE", str(round(shake_metrics["accuracy"][n], 4)), str(round(shake_metrics["precision"][n], 4)),
         str(round(shake_metrics["recall"][n], 4)), str(round(shake_metrics["f1"][n], 4)), ],
        ["TWIST", str(round(twist_metrics["accuracy"][n], 4)), str(round(twist_metrics["precision"][n], 4)),
         str(round(twist_metrics["recall"][n], 4)), str(round(twist_metrics["f1"][n], 4)), ],

    ]

    return data


def metrics_calc(origin, metrics_dest, number):

    tp = origin["true_positive"][number]
    fp = origin["false_positive"][number]
    fn = origin["false_negative"][number]
    tn = origin["true_negative"][number]

    metric_accuracy = (tp + tn) / (fp + fn + tp + tn)
    metric_recall = tp / (fn + tp)
    metric_precision = tp / (fp + tp)
    metric_f1 = 2 * (metric_precision * metric_recall) / (metric_precision + metric_recall)

    metrics_dest["accuracy"].append(metric_accuracy)
    metrics_dest["recall"].append(metric_recall)
    metrics_dest["precision"].append(metric_precision)
    metrics_dest["f1"].append(metric_f1)


def simple_metrics_calc(origin, metrics_dest):
    tp = origin["true_positive"]
    fp = origin["false_positive"]
    fn = origin["false_negative"]
    tn = origin["true_negative"]

    metric_accuracy = (tp + tn) / (fp + fn + tp + tn)
    metric_recall = tp / (fn + tp)
    metric_precision = tp / (fp + tp)
    metric_f1 = 2 * (metric_precision * metric_recall) / (metric_precision + metric_recall)

    metrics_dest["accuracy"] = metric_accuracy
    metrics_dest["recall"] = metric_recall
    metrics_dest["precision"] = metric_precision
    metrics_dest["f1"] = metric_f1


def filling_table(dict, header):
    data = [
        [header, "Mean", "Std Dev", "Max", "Min", ],
        ["True Positive", str(round(statistics.mean(dict["true_positive"]), 2)), str(round(statistics.stdev(dict["true_positive"]), 2)),
         str(max(dict["true_positive"])), str(min(dict["true_positive"])), ],
        ["False Positive", str(round(statistics.mean(dict["false_positive"]), 2)), str(round(statistics.stdev(dict["false_positive"]), 2)),
         str(max(dict["false_positive"])), str(min(dict["false_positive"])), ],
        ["False Negative", str(round(statistics.mean(dict["false_negative"]), 2)), str(round(statistics.stdev(dict["false_negative"]), 2)),
         str(max(dict["false_negative"])), str(min(dict["false_negative"])), ],
        # ["True Negative", str(round(statistics.mean(dict["true_negative"]), 2)), str(round(statistics.stdev(dict["true_negative"]), 2)),
        #  str(max(dict["true_negative"])), str(min(dict["true_negative"])), ],
    ]

    return data


class BahdanauAttention(Layer):
    def __init__(self, units, verbose=0):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
        self.verbose = verbose

    def call(self, query, values):
        if self.verbose:
            print('\n******* Bahdanau Attention STARTS******')
            print('query (decoder hidden state): (batch_size, hidden size) ', query.shape)
            print('values (encoder all hidden state): (batch_size, max_len, hidden size) ', values.shape)

        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        if self.verbose:
            print('query_with_time_axis:(batch_size, 1, hidden size) ', query_with_time_axis.shape)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))
        if self.verbose:
            print('score: (batch_size, max_length, 1) ', score.shape)
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        if self.verbose:
            print('attention_weights: (batch_size, max_length, 1) ', attention_weights.shape)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        if self.verbose:
            print('context_vector before reduce_sum: (batch_size, max_length, hidden_size) ', context_vector.shape)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        if self.verbose:
            print('context_vector after reduce_sum: (batch_size, hidden_size) ', context_vector.shape)
            print('\n******* Bahdanau Attention ENDS******')
        return context_vector, attention_weights


def training_encoder_decoder(out_dim, input_params, out_labels, start_n, batch_s, time_ss):

    # The first part is encoder
    encoder_inputs = Input(shape=(None, input_params), name='encoder_inputs')
    encoder_lstm = LSTM(out_dim, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)

    # initial context vector is the states of the encoder
    encoder_states = [encoder_state_h, encoder_state_c]

    # Set up the attention layer
    attention = BahdanauAttention(out_dim)

    # Set up the decoder layers
    decoder_inputs = Input(shape=(1, (out_labels + out_dim)), name='decoder_inputs')
    decoder_lstm = LSTM(out_dim, return_state=True, name='decoder_lstm')
    decoder_dense = Dense(out_labels, activation='softmax', name='decoder_dense')

    all_outputs = []

    # 1 initial decoder's input data
    # Prepare initial decoder input data that just contains the start character
    # Note that we made it a constant one-hot-encoded in the model
    # that is, [1 0 0 0 0 0 0 0 0 0] is the first input for each loop
    # one-hot encoded zero(0) is the start symbol

    inputs = np.zeros((batch_s, 1, out_labels), dtype="float32")
    # inputs = np.zeros((1, 1, out_labels), dtype="float32")
    inputs[:, :, :] = start_n
    # inputs[:, 0, 0] = start_n

    # 2 initial decoder's state
    # encoder's last hidden state + last cell state
    decoder_outputs = encoder_state_h
    states = encoder_states

    # decoder will only process one time step at a time.
    for _ in range(time_ss):
        # 3 pay attention
        # create the context vector by applying attention to
        # decoder_outputs (last hidden state) + encoder_outputs (all hidden states)
        context_vector, attention_weights = attention(decoder_outputs, encoder_outputs)

        context_vector = tf.expand_dims(context_vector, 1)
        # context_vector = tf.expand_dims(context_vector, batch_s)

        # 4. concatenate the input + context vector to find the next decoder's input
        inputs = tf.concat([context_vector, inputs], axis=-1)

        # 5. passing the concatenated vector to the LSTM
        # Run the decoder on one timestep with attended input and previous states
        decoder_outputs, state_h, state_c = decoder_lstm(inputs,
                                                         initial_state=states)
        # decoder_outputs = tf.reshape(decoder_outputs, (-1, decoder_outputs.shape[2]))

        outputs = decoder_dense(decoder_outputs)
        # 6. Use the last hidden state for prediction the output
        # save the current prediction
        # we will concatenate all predictions later
        outputs = tf.expand_dims(outputs, 1)
        # outputs = tf.expand_dims(outputs, 2)
        all_outputs.append(outputs)
        # 7. Reinject the output (prediction) as inputs for the next loop iteration
        # as well as update the states
        inputs = outputs
        states = [state_h, state_c]

    # 8. After running Decoder for max time steps
    # we had created a predition list for the output sequence
    # convert the list to output array by Concatenating all predictions
    # such as [batch_size, timesteps, features]

    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

    # 9. Define and compile model
    model = Model(encoder_inputs, decoder_outputs, name='model_encoder_decoder')
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

    return model

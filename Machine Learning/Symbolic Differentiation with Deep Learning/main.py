import re
import time
import json
import numpy as np
from tensorflow import keras
import tensorflow as tf

LSTM_UNITS = 750
MAX_SEQUENCE_LENGTH = 32


def predict_derivative(function_vector):
    states_value = encoder_model.predict(function_vector, verbose=0)
    derivative_vector = np.zeros((1, 1, num_tokens))
    derivative_vector[0, 0, token_to_index["\t"]] = 1.0

    stop_condition = False
    predicted_derivative = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([derivative_vector] + states_value, verbose=0)
        predicted_derivative_token_index = np.argmax(output_tokens[0, -1, :])
        predicted_derivative_token = index_to_token[str(predicted_derivative_token_index)]
        predicted_derivative += predicted_derivative_token

        if predicted_derivative_token == "\n" or len(predicted_derivative) > MAX_SEQUENCE_LENGTH:
            stop_condition = True

        derivative_vector = np.zeros((1, 1, num_tokens))
        derivative_vector[0, 0, predicted_derivative_token_index] = 1.0

        states_value = [h, c]
    return predicted_derivative.strip().replace('/t', '').replace('/n', '')


if __name__ == "__main__":

    print("Loading trained model .....")
    model = keras.models.load_model("s2s_990000_latent_750_cat_cross_etrp")

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    decoder_inputs = tf.identity(decoder_inputs)
    decoder_state_input_h = keras.Input(shape=(LSTM_UNITS,))
    decoder_state_input_c = keras.Input(shape=(LSTM_UNITS,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )
    print('Model loaded successfully!!')

    with open('token_to_index.json', 'r') as f:
        token_to_index = json.load(f)
    with open('index_to_token.json', 'r') as f:
        index_to_token = json.load(f)

    num_tokens = len(token_to_index)
    filepath_test = "test.txt"
    num_samples_test = 100
    functions_test = []
    derivatives_test = []
    with open(filepath_test, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    for line in list(filter(None, lines[: min(num_samples_test, len(lines))])):
        function, derivative = line.split("=")
        functions_test.append(function)
        derivatives_test.append(derivative)

    print("Number of test samples:", len(functions_test))
    print("Predicting derivatives for test file .......")
    encoder_input_data_test = np.zeros((len(functions_test), MAX_SEQUENCE_LENGTH, num_tokens), dtype="float32")

    for i, function in enumerate(functions_test):
        for t, token in enumerate(
                list(filter(None, re.split(r"(exp|sin|cos|\^|[a-zA-Z]|[0-9]|[()]|\+|\*|\n|\t)", function)))):
            encoder_input_data_test[i, t, token_to_index[token]] = 1.0
        encoder_input_data_test[i, t + 1:, token_to_index['']] = 1.0

    match = 0
    start = time.time()
    for function_index in range(num_samples_test):
        input_function_vector = encoder_input_data_test[function_index: function_index + 1]
        decoded_sentence = predict_derivative(input_function_vector)
        end = time.time()
        if decoded_sentence == derivatives_test[function_index]:
            match += 1
        else:
            print("Wrong prediction !! Test function {}, Predicted derivative {}, Actual derivative {}"
                  .format(functions_test[function_index], decoded_sentence, derivatives_test[function_index]))
        if (function_index + 1) % 10 == 0:
            print("{} predicted_test_derivatives matched with {} actual_test_derivatives in total {}s"
                  .format(match, function_index + 1, int(end - start)))

    print("accuracy is : ", match / num_samples_test)

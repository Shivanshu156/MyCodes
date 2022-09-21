import re
import json
import numpy as np
from tensorflow import keras

MAX_SEQUENCE_LEN = 32
BATCH_SIZE = 512
EPOCHS = 10
LSTM_UNITS = 750
NUM_SAMPLES = 1000000
filepath = "train.txt"


def tokenize(expression):
    return list(filter(None, re.split(r"(exp|sin|cos|\^|[a-zA-Z]|[0-9]|[()]|\+|\*|\n|\t)", expression)))


if __name__ == "__main__":
    functions = list()
    derivatives = list()
    tokens = set()

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    for line in list(filter(None, lines[: min(NUM_SAMPLES, len(lines))])):
        function, derivative = line.split("=")
        derivative = "\t" + derivative + "\n"  # using /t and /n as start and end tokens
        functions.append(function)
        derivatives.append(derivative)
        for token in tokenize(function):
            if token not in tokens:
                tokens.add(token)

    tokens.add('\n')
    tokens.add('\t')
    tokens.add('')
    tokens = sorted(list(tokens))
    num_tokens = len(tokens)

    print("Number of samples:", len(functions))
    print("Number of unique tokens:", num_tokens)
    print("Max sequence length:", MAX_SEQUENCE_LEN)

    token_to_index = dict([(token, i) for i, token in enumerate(tokens)])
    index_to_token = dict((i, token) for token, i in token_to_index.items())
    with open("token_to_index.json", "w") as fp:
        json.dump(token_to_index, fp)
    with open("index_to_token.json", "w") as fp:
        json.dump(index_to_token, fp)
    print('Token dictionary saved')
    encoder_input_data = np.zeros((len(functions), MAX_SEQUENCE_LEN, num_tokens), dtype="float32")
    decoder_input_data = np.zeros((len(functions), MAX_SEQUENCE_LEN, num_tokens), dtype="float32")
    decoder_target_data = np.zeros((len(functions), MAX_SEQUENCE_LEN, num_tokens), dtype="float32")

    for i, (function, derivative) in enumerate(zip(functions, derivatives)):

        for t_f, token_f in enumerate(tokenize(function)):
            encoder_input_data[i, t_f, token_to_index[token_f]] = 1.0
        encoder_input_data[i, t_f + 1:, token_to_index['']] = 1.0

        for t_d, token_d in enumerate(tokenize(derivative)):
            decoder_input_data[i, t_d, token_to_index[token_d]] = 1.0
            if t_d > 0:
                decoder_target_data[i, t_d - 1, token_to_index[token_d]] = 1.0
        decoder_input_data[i, t_d + 1:, token_to_index['']] = 1.0
        decoder_target_data[i, t_d:, token_to_index['']] = 1.0

    print('function and derivative vectors created successfully!!')

    # Build the model

    encoder_inputs = keras.Input(shape=(None, num_tokens))
    encoder = keras.layers.LSTM(LSTM_UNITS, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = keras.Input(shape=(None, num_tokens))
    decoder_lstm = keras.layers.LSTM(LSTM_UNITS, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(num_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Train the model
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
    )

    # Save model
    model.save("model_{}_{}_{}".format(len(functions), LSTM_UNITS, "cat_cross_etrp"))
    print("Model saved successfully!!")

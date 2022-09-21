Derivatives using Deep Learning


Problem Statement: 

Create a deep learning model that learns to take the derivative of the function with respect to the requested variable.
•	sample: d(6exp^(9a)+4exp^(2a))/da=54exp^(9a)+8exp^(2a)
•	6exp^(9a)+4exp^(2a) is the function
•	d(...)/da means "take the derivative of the function with respect to a"
•	54exp^(9a)+8exp^(2a) is the derivative of the input function

Approach:

Given a mathematical expression, finding its derivative classifies into a sequence to sequence problem. As the sequence of characters in the string of function is very important here, I went for a sequence to sequence approach.

In deep learning, RNNs perform very well for sequential problems but they have short memory due to gradient vanishing problem. LSTMs as named Long Short-Term Memory Networks, are capable of holding memory longer and that’s why I have chosen LSTMs for this problem.

Tokenization: 

For a given sample, I can tokenize a string using its characters, but this can confuse the network. For e.g, tokenizing operators like sin will give a list as [‘s’, ‘i’, ‘n’] and if the given differentiation is w.r.t variable n, then the embedding for sin will also contain the embedding of char n. Therefore, I have extracted the operators as a whole and considered all the operators as a single token. I did this using regular expression operations in Python. 

def tokenize(expression):
    return list(filter(None, re.split(r"(exp|sin|cos|\^|[a-zA-Z]|[0-9]|[()]|\+|\*|\n|\t)", expression)))

The above implementation will consider exp, cos and sin operators as a single token. Also, I could use one hot encoding here, but one hot encoding gives orthogonal vectors and here tokens are ordinal, so preferred Label Encoding. 

Many mathematical expression problems convert expressions  into prefix notations, however, as parenthesis are a part of sequence here and the maximum sequence length has a limit of 30, I didn’t convert it to prefix notation.

Network:

•	An encoder LSTM turns functions(input sequence) to 2 state vectors 
•	A decoder LSTM is trained to turn the target sequences(derivatives) into the same sequence but offset by one timestep in the future, a training process called "teacher forcing" in this context. It uses as initial state the state vectors from the encoder. Effectively, the decoder learns to generate `targets[t+1...]` given `targets[...t]`, conditioned on the input sequence.


•	In inference mode, when we want to decode unknown input sequences, we:
-	Encode the input sequence into state vectors
-	Start with a target sequence of size 1 (just the start-of-sequence token)
-	Feed the state vectors and 1-token target sequence to the decoder to produce predictions for the next token.
-	Sample the next token using these predictions (simply used argmax).
-	Append the sampled token to the target sequence
-	Repeat until we generate the end-of-sequence token(‘\n’) or we hit the sequence limit.
•	As the trainable parameters limit is 5M, used 750 latent dimension which produced 4.8M params.

References:

1.	Deep Learning for Symbolic Mathematics
2.	https://github.com/facebookresearch/SymbolicMathematics
3.	https://github.com/keras-team/keras-io/blob/master/examples/nlp/lstm_seq2seq.py


Licence:
This is a licensed problem with Scale AI copyright. This code is written by me with reference to above references.

Copyright © Scale AI 2022 - All Rights Reserved

This takehome is the property of Scale AI. Copying, distributing or sharing this takehome and any derivative works created from this takehome via any medium is strictly prohibited. Any attempt to do so is a violation of the rights of Scale AI. Individuals who breach these terms may be subject to prosecution and damages.

Proprietary and confidential
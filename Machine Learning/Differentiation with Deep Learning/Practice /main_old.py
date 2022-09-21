# import re
# import numpy as np
# from typing import Tuple
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
#
# MAX_SEQUENCE_LENGTH = 30
# TRAIN_URL = "https://drive.google.com/file/d/1ND_nNV5Lh2_rf3xcHbvwkKLbY2DmOH09/view?usp=sharing"
#
#
# def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
#     """loads the test file and extracts all functions/derivatives"""
#     data = open(file_path, "r").readlines()
#     functions, derivatives = zip(*[line.strip().split("=") for line in data])
#     return functions, derivatives
#
#
# def score(true_derivative: str, predicted_derivative: str) -> int:
#     """binary scoring function for model evaluation"""
#     return int(true_derivative == predicted_derivative)
#
#
# # --------- PLEASE FILL THIS IN --------- #
# def predict(functions: str):
#     return functions
#
#
# # ----------------- END ----------------- #
#
# def get_label_encoder(functions, true_derivatives):
#     charset = set()
#     for function in functions:
#         for x in re.split(r"(exp|sin|cos|\^|\/|[a-zA-Z]|[0-9]|[()]|\+|\-|\*|\#)", function):
#             charset.add(x)
#     label_encoder = LabelEncoder()
#     return label_encoder.fit(list(charset))
#
#
# def main(filepath: str = "test.txt"):
#     """load, inference, and evaluate"""
#     functions, true_derivatives = load_file(filepath)
#     predicted_derivatives = [predict(f) for f in functions]
#     scores = [score(td, pd) for td, pd in zip(true_derivatives, predicted_derivatives)]
#     print(np.mean(scores))
#
#
# def train(filepath: str = "train.txt"):
#     """load, inference, and evaluate"""
#     functions, true_derivatives = load_file(filepath)
#     functions = [function.ljust(30, "#") for function in functions]
#     true_derivatives = [true_derivative.ljust(30, "#") for true_derivative in true_derivatives]
#     label_encoder = get_label_encoder(functions, true_derivatives)
#
#
#
# if __name__ == "__main__":
#     train()
#     # main()

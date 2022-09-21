# NLP take-home: Derivatives

### Problem Statement

Create a deep learning model that learns to take the derivative of the function with respect to the requested variable. 
This exercised is aimed to allow candidates to demonstrate their machine learning prowess, and as such, please limit
using heuristics or rules.

A training file is provided in Google Drive:
* `train.txt`: https://drive.google.com/file/d/1WJklLePf9uLlFMg_IrDRm5Hm7phOh6cX/view?usp=sharing

Each line of `train.txt` contains one training sample with the groundtruth result. 
The model should take the original function, and predict the derivative.

* sample: `d(6exp^(9a)+4exp^(2a))/da=54exp^(9a)+8exp^(2a)`
* `6exp^(9a)+4exp^(2a)` is the function
* `d(...)/da` means "take the derivative of the function with respect to `a`"
* `54exp^(9a)+8exp^(2a)` is the derivative of the input function

While the final derivative is commutable, only exact string matches are considered correct.

### Deliverables

Please submit a zip file with the following included:

1. The final score on a portion of the train.txt file
2. The trained model
3. A `network.txt` file that summarizes the model architecture with the number of trainable parameters at each layer
  * For TensorFlow, a simple `model.summary()` will suffice.
  * For pytorch, consider [`torchsummary`](https://github.com/sksq96/pytorch-summary).
4. The training code in `train.py`  which reproduces the final model submitted
5. A working `main.py` that evaluates the score of the trained model on a blind `test.txt`
  * `test.txt` will follow the exact same pattern as `train.txt`
6. A `requirements.txt` file that includes all python dependencies and their versions
7. (Optional) A brief explanation of your approach

### Evaluation

Model performance will be evaluated on a blind `test.txt` using `main.py`. Graders will create a fresh 
virtual environment using a python environment > 3.6 and execute main.py using the following commands
```
conda create -y -n homeworkenv
conda activate homeworkenv
cd $HOMEWORK_CODE_DIRECTORY
pip install -r requirements.txt
python main.py  
``` 

after which graders will expect to see the mean accuracy score on the test set. 

Submissions will be graded by the below criteria:
   * Performance on the blind test set. The minimum passing score is 0.7 but we recommend pushing the model as far as possible.
   * Model design choices (e.g. network size, architecture, regularization)
   * Python proficiency and implementation details

Passing the minimum score threshold is not a guarantee of acceptance, all three factors will be considered for evaluation.	 

### Tips
* Google Colab offers free GPU compute
* The maximum input or output sequence length is 30.
* Unnecessarily large models (>5M parameters) will be penalized.


### License
Copyright © Scale AI 2022 - All Rights Reserved

This takehome is the property of Scale AI. Copying, distributing or sharing this takehome 
and any derivative works created from this takehome via any medium is strictly prohibited. 
Any attempt to do so is a violation of the rights of Scale AI. Individuals who breach these 
terms may be subject to prosecution and damages.

Proprietary and confidential
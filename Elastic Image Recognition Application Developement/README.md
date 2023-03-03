# CSE 546 Project 1 

Daksh Dobhal, Shivanshu Verma, Gaurav Verma
## Dependencies

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.

```bash
pip3 install flask
pip3 install boto3
pip3 install json
pip3 install os
pip3 install werkzeug
pip3 install subprocess
pip3 install ast
pip3 install yaml
pip3 install pillow
```

## Usage

```python
python multithread_workload_generator.py --num_request 100 --url 'http://3.82.92.105/upload-image' --image_folder "PATH/TO/IMAGE_FOLDER/imagenet-100/"

```

## URLs of AWS Resources


1. Input S3 Bucket : inputapptierbucket
2. Output S3 Bucket : outputapptierbucket
3. Request SQS Queue : https://sqs.us-east-1.amazonaws.com/920241055366/RequestQueue
4. Response SQS Queue : https://sqs.us-east-1.amazonaws.com/920241055366/ResponseQueue

## License

[MIT](https://choosealicense.com/licenses/mit/)

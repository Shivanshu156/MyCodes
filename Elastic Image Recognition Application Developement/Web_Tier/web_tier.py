from flask import Flask, request
import boto3
import json
import io
from PIL import Image

app = Flask(__name__)
s3 = boto3.client('s3')
sqs = boto3.resource('sqs', region_name='us-east-1')

# Request and Response Queue
request_queue = sqs.get_queue_by_name(QueueName='RequestQueue')
response_queue = sqs.get_queue_by_name(QueueName='ResponseQueue')

# Input and Output S3 Buckets
INPUT_BUCKET_NAME = "inputapptierbucket"
OUTPUT_BUCKET_NAME = "outputapptierbucket"

# Startup Page
@app.route('/')
def home():
    return "App Started"

# Main functionality of Web-Tier
@app.route('/upload-image', methods=['POST'])
def upload_image():

    # Extracting Image details from request recieved.
    file = request.files['myfile']
    filename = file.filename
    file_contents = file.read()
    image = Image.open(io.BytesIO(file_contents))
    jpeg_image = io.BytesIO()
    image.save(jpeg_image, 'JPEG')
    jpeg_data = jpeg_image.getvalue()

    # Uploading image to S3
    s3.put_object(Body=jpeg_data, Bucket=INPUT_BUCKET_NAME, Key=filename)

    # Sending message on Request Queue for App-Tier to classify
    message = {'filename': filename}
    request_queue.send_message(MessageBody=json.dumps(message))

    # Waiting for notification on Response Queue
    while True:
        # Receive messages from the Response Queue
        messages = response_queue.receive_messages(MaxNumberOfMessages=10, WaitTimeSeconds=1)
        for message in messages:
            # Checking if file has been classified or not.
            if filename in message.body:
                s3_response = s3.get_object(Bucket = OUTPUT_BUCKET_NAME, Key = filename[:-5])
                # Reading text_stream from S3.
                file_stream = s3_response['Body']
                txt = str(file_stream.read())
                filename, ans = txt.split(',')
                # For debugging purpose.
                # print(filename[3:])
                # print(ans[:-4])

                # Deleting message from Response Queue.
                k = message.delete()
                # Returning classification result to user.
                return filename[3:] + ".JPEG : " + ans[:-4]

if __name__ == "__main__":
    app.run(threaded = True)

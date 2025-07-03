import boto3
import json
import base64
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def main():
    # Get credentials from environment
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "ap-south-1")

    # Initialize Bedrock client
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=aws_region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

    # Model and prompt config
    model_id = "amazon.titan-image-generator-v1"
    prompt = "A romantic couple sitting on a bench in a peaceful park, holding hands, surrounded by trees and flowers, soft sunlight, cinematic atmosphere"
    accept = "application/json"
    content_type = "application/json"

    # Titan request body format
    body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 512,
            "width": 512,
            "seed": 0,
            "cfgScale": 10
        }
    })

    # Call Titan model
    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=body,
        accept=accept,
        contentType=content_type
    )

    # Parse response
    response_body = json.loads(response['body'].read())
    base64_image = response_body['images'][0]

    # Decode and display/save image
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))
    image.show()
    image.save("titan_output.png")  # optional: save image

if __name__ == "__main__":
    main()

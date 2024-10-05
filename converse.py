import base64
import os
from dotenv import load_dotenv
load_dotenv()

import requests


def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def converse(image_path):
    endpoint = "https://openai-eastus-instance-02.openai.azure.com/"
    api_key = os.getenv('GPT_API_KEY')
    deployment_name = "gpt-4o"
    api_version = "2024-02-15-preview"

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    image_base64 =  get_image_base64(image_path)
    
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that generates captions for images. When presented with an image, analyze it and provide a relevant caption."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please provide a short explanation for the images.\
                    The content should be relevant with all the details in the image"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }
    ]

    payload = {
        "messages": messages,
        "max_tokens": 800,
        "temperature": 0.2,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "top_p": 0.95
    }

    response = requests.post(
        f"{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        response_data = response.json()
        caption = response_data["choices"][0]["message"]["content"].strip()
        return caption
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")
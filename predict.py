# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

import requests
import time
import json

import base64
from PIL import Image
from io import BytesIO

class Prodia:
    def __init__(self, api_key, base=None):
        self.base = base or "https://api.prodia.com/v1"
        self.headers = {
            "X-Prodia-Key": api_key
        }
    
    def generate(self, params):
        response = self._post(f"{self.base}/job", params)
        return response.json()
    
    def transform(self, params):
        response = self._post(f"{self.base}/transform", params)
        return response.json()
    
    def controlnet(self, params):
        response = self._post(f"{self.base}/controlnet", params)
        return response.json()
    
    def get_job(self, job_id):
        response = self._get(f"{self.base}/job/{job_id}")
        return response.json()

    def wait(self, job):
        job_result = job

        while job_result['status'] not in ['succeeded', 'failed']:
            time.sleep(0.25)
            job_result = self.get_job(job['job'])

        return job_result

    def list_models(self):
        response = self._get(f"{self.base}/models/list")
        return response.json()

    def _post(self, url, params):
        headers = {
            **self.headers,
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, data=json.dumps(params))

        if response.status_code != 200:
            raise Exception(f"Bad Prodia Response: {response.status_code}")

        return response

    def _get(self, url):
        response = requests.get(url, headers=self.headers)

        if response.status_code != 200:
            raise Exception(f"Bad Prodia Response: {response.status_code}")

        return response


def image_to_base64(image_path):
    # Open the image with PIL
    with Image.open(image_path) as image:
        # Convert the image to bytes
        buffered = BytesIO()
        image.save(buffered, format="PNG")  # You can change format to PNG if needed
        
        # Encode the bytes to base64
        img_str = base64.b64encode(buffered.getvalue())

    return img_str.decode('utf-8')  # Convert bytes to string

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        api_key: str = Input(
            description="Prodia API Key"
        ),
        prompt: str = Input(
            description="Prompt", default="puppies in a cloud, 4k"
        ),
        model: str = Input(
            description="Model", default="v1-5-pruned-emaonly.safetensors [d7049739]"
        ),
        negative_prompt: str = Input(
            description="Negative Prompt", default="badly drawn"
        ),
        steps: int = Input(
            description="Steps", default=25
        ),
        cfg_scale: int = Input(
            description="CFG Scale", default=7
        ),
        sampler: str = Input(
            description="Sampler", default="DPM++ 2M Karras"
        )
    ) -> Path:
        """Run a single prediction on the model"""
        prodia_client = Prodia(api_key=api_key)

        result = prodia_client.generate({
            "model": model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler": sampler
        })

        job = prodia_client.wait(result)

        return job["imageUrl"]
# -*- coding:utf-8 -*-

import requests
from io import BytesIO
import base64
import json
from PIL import Image

url = 'http://127.0.0.1:5000/api/predict'
headers = {"Content-Type": "application/json"}
image = Image.open('/Users/ethan/MyCodes/baidu_tensor/test/1698.jpg')


def image_to_string(image):
    out = BytesIO()
    image.save(out, format='JPEG')

    str = base64.b64encode(out.getvalue()).decode('utf-8')
    s = json.dumps({'image': str})
    r = requests.post(url, data=s, headers=headers)

    print(r.json()['y'])

def main():
    image_to_string(image)

if __name__ == '__main__':
    main()
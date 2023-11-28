from flask import Flask, jsonify, request, redirect, url_for
import numpy as np
import io
import os
import json
from flask_cors import CORS
import re
import torch
from PIL import Image
from validacion import test_pipeline

app = Flask(__name__)
CORS(app)

#URL_BASE = "https://shaggy-roses-arrive.loca.lt"
URL_BASE = "https://oxxo-object-detect.loca.lt"
PLAN_DICT = {}
REALOGRAM = np.empty(0)
PLAN_ID = -1
result = {}
EXT = ''
pattern = re.compile(r'Planograma (\d+)')

def reload_plans():
    global PLAN_DIR, PLAN_DICT

    image_list = []
    # TODO: where are we pulling the plans from
    
    for f in os.listdir(PLAN_DIR):
        # Separa la extensión y el nombre base
        base_name, extension = os.path.splitext(f)

        # Verifica si el nombre base coincide con el patrón
        if pattern.match(base_name) and extension.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
            image_list.append(f)

    image_list.sort(key=lambda x: int(
        pattern.match(os.path.splitext(x)[0]).group(1)
    ) # sort by the number after "gondola." in the filename
    )

    # overwrite plans
    PLAN_DICT = {i: filename for i, filename in enumerate(image_list,1)}


def process_real(id: int, real_img: np.array):
    global model

    # classify realogram
    real_boxes = model([real_img]).xywhn[0]

    # classify planogram
    plan_boxes = model([Image.open(os.path.join(PLAN_DIR, PLAN_DICT[id]))]).xywhn[0]
    
    if plan_boxes.any() > 0 and real_boxes.any() > 0:
        plan_boxes = np.c_[
            range(len(plan_boxes)),
            (plan_boxes[:, [0,1,2,3]]).numpy(),
            plan_boxes[:, -1]
        ]

        last_idx = int(plan_boxes[-1, 0])
        
        real_boxes = np.c_[
            range(last_idx, last_idx+len(real_boxes)),
            (real_boxes[:, [0,1,2,3]]).numpy(),
            real_boxes[:, -1]
        ]
        

        # validate
        processed_image_name, score, errors = test_pipeline(plan_boxes, real_boxes, model.names, real_img=real_img, save_dir=REAL_DIR)
        
        return processed_image_name, score, errors


@app.get('/get-plans')
def get_plans():
    reload_plans()
    # construct image info as id, filename (no extension), image redirect
    image_data = []
    for i, filename in PLAN_DICT.items():
        name, ext = filename.rsplit('.', 1)
        image_data.append({
            'id': i,
            'name': name,
            'image': URL_BASE + url_for('static', filename=filename)
        })

    return jsonify({
        "data":
            {
                "plans":
                image_data
            }
    })


@app.post('/post-real')
def post_real():
    """Incoming requests should look like:
    {
            "data":
            {
                "image": ???,
                "plan-id": 1
            }
        }

        Returns JSON like:
    {
        "data":
        {
            "image": ???,
            "score": 0.98227931096978203,
            "errors":
            [
                "En {pos} se detectó {prod1} en lugar de {prod2}",
                "En {pos} se detectó {prod} sobrante",
                "En {pos} se detectó {prod} faltante"
            ]
        }
    }
    """
    global REALOGRAM, PLAN_ID, EXT
    
    file = request.files.get('image', None)
    if file is None:
        return redirect('/')

    REALOGRAM = Image.open(io.BytesIO(file.read()))
    PLAN_ID = int(request.form.get('plan-id', -1))
    
    if PLAN_ID == -1:
        return redirect('/')

    processed_image_name, score, errors = process_real(PLAN_ID, REALOGRAM)
    
    return jsonify({
        "data":
        {
            'image': URL_BASE + url_for('static', filename=processed_image_name),
            "score": score,
            "errors": errors
        }
    })


@app.route('/')
def home():
    return 'Home Page'


if __name__ == '__main__':
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
        PLAN_DIR = config.get('image_directory', os.getcwd())
        REAL_DIR = config.get('result_directory', os.getcwd())

    # load model
    model = torch.hub.load('yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt', source='local', verbose=0)
    model.eval()

    reload_plans()
    app.run(host='0.0.0.0', port=5000)

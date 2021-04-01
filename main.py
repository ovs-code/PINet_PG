from flask import Flask, url_for, render_template, request, session, redirect, flash, jsonify, send_file
from markupsafe import escape
from werkzeug.utils import secure_filename
import os
import sys
import torch
import io
import base64

# import asyncio
# from mlq.queue import MLQ

from PIL import Image

from pipeline import InferencePipeline
from options.infer_options import InferOptions
from tool import cords_to_map, get_coords, reorder_pose, load_pose_from_file

# necessary variable for using tf and torch simulatuosly
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# read inference options and create new pipeline
# opt = InferOptions().parse(['--name', 'fashion_PInet_PG', '--pose_estimator', 'assets/pretrains/pose_estimator.h5'])
# pipeline = InferencePipeline.from_opts(opt=opt)

# set the directory for the templates
template_dir = os.path.abspath('./webapp/templates')
static_dir = os.path.abspath('./webapp/static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Set the secrete key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Set the upload folder for the user images
UPLOAD_FOLDER = 'webapp/static/imgs'

# for testing purpose only set the path to the target poses
TARGET_POSE_PATH = 'test_data/testK/randomphoto_small.jpg.npy'
# directly load the target pose
target_pose = load_pose_from_file(TARGET_POSE_PATH)

# Set the allowed extensions (i.e. only images)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# the poses one can select from
CROSSED_LEG_POSE = list(zip([49, 43, 43, 49, 49, 87, 83, 129, 121, 155, 153, 149, 151, 151, 151, 177, 179], [97, 103, 93, 111, 85, 123, 71, 123, 61, 99, 71, 107, 73, 155, 21, 85, 93]))

# dummy image for testing the server
DUMMY_OUTPUT_IMAGE = Image.open('./test_data/out.jpg')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# create an instance for the ml queue for handling the inference at the backend
#mlq = MLQ(q_name='pose_transfer', redis_host='localhost', redis_port=6379, redis_db=0)
#
#CALLBACK_URL = 'http://localhost:3000/callback'
#

## Routes for MLQ (GPU server)
#@app.route('/do_transfer', methods=['POST'])
#def do_pose_transfer():
#    # request the backend to transfer the incoming image
#    # but for now this is just a dummy for testing mlq
#    assert request.json['number']
#    job_id = mlq.post(request.json, CALLBACK_URL)
#    return jsonify({'msg': 'Processing, check back soon.', 'job_id': job_id})
#
#@app.route('/status/<job_id>', methods=['GET'])
#def get_progress(job_id):
#    return jsonify({'msg': mlq.get_progress(job_id=job_id)})
#
#@app.route('/result/<job_id>', methods=['GET'])
#def get_result(job_id):
#    job = mlq.get_job(job_id=job_id)
#    return jsonify({'short_result': job['short_result']})
#
#@app.route('/callback', methods=['GET'])
#def train_model():
#    success = request.args.get('success', None)
#    job_id = request.args.get('job_id', None)
#    short_result = request.args.get('short_result', None)
#    print('We receied a callback! Job ID {} returned sucessful={} with short_esult{}'.format(
#        job_id, success, short_result
#    ))
#    return 'ok'
# check for allowed extensions only
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# app routes for serving the website
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print('POST request received')
        if 'image' not in request.files or not request.files['image']:
            flash('No image selected')
            return redirect(request.url)
        image = request.files['image']
        if not allowed_file(image.filename):
            flash(f'Image Type has to be one of {ALLOWED_EXTENSIONS}')
            return redirect(request.url)
        
        source_image = Image.open(image).convert(mode='RGB')
        img_width, img_height = source_image.size
        if img_width != 176 or img_height != 256:
            flash('Image has wrong input format. Try again.')
            return redirect(request.url)
        
        # generate the output image
        # output_image = pipeline(image=source_image, target_pose_map=target_pose)
        output_image = DUMMY_OUTPUT_IMAGE

        # create file-object in memory to store output
        file_obj = io.BytesIO()

        # write image in file-object
        output_image.save(file_obj, 'PNG')

        # get the bytestream from the file
        img_data = file_obj.getvalue()
        # encode the bytestream to base64
        img_data = base64.b64encode(img_data)

        return {
            'output_image': img_data.decode()
        }

    elif request.method == 'GET':
        pose = CROSSED_LEG_POSE
        poses = [pose]
        return render_template('index.html', poses=poses)
    else:
        return f'error {request.method} method not implemented'

@app.route('/impressum')
def impressum():
    return render_template('impressum.html')

with app.test_request_context():
    print(url_for('index'))
    print(url_for('impressum'))

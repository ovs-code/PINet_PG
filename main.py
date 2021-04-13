from flask import Flask, url_for, render_template, request, session, redirect, flash, jsonify, send_file
from markupsafe import escape
from werkzeug.utils import secure_filename
import os
import sys
import torch
import io
import base64

import asyncio
from mlq.queue import MLQ

from PIL import Image

from pipeline import InferencePipeline
from options.infer_options import InferOptions
# from tool import cords_to_map, get_coords, reorder_pose, load_pose_from_file

# necessary variable for using tf and torch simulatuosly
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
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

# Set the allowed extensions (i.e. only images)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# the poses one can select from
# CROSSED_LEG_POSE = list(zip([49, 43, 43, 49, 49, 87, 83, 129, 121, 155, 153, 149, 151, 151, 151, 177, 179], [97, 103, 93, 111, 85, 123, 71, 123, 61, 99, 71, 107, 73, 155, 21, 85, 93]))
POSES = [[[205, 100], [209, 99], [207, 104], [205, 93], [205, 110], [189, 86], [189, 115], [215, 78], [211, 126], [239, 78], [235, 128], [133, 89], [133, 106], [95, 88], [95, 97], [61, 86], [61, 91]], [[43, 97], [37, 97], [39, 93], [43, 80], [41, 82], [67, 82], [67, 78], [101, 95], [103, 97], [77, 100], [79, 102], [133, 91], [135, 84], [183, 82], [181, 82], [229, 80], [231, 80]], [[121, 84], [117, 88], [117, 78], [121, 93], [121, 71], [149, 104], [147, 60], [181, 115], [183, 51], [195, 126], [193, 36], [205, 97], [203, 64], [211, 133], [211, 31], [229, 71], [231, 71]], [[43, 91], [39, 95], [39, 88], [41, 97], [43, 77], [69, 86], [67, 78], [99, 95], [101, 97], [75, 102], [79, 102], [133, 91], [137, 82], [183, 80], [183, 80], [229, 78], [229, 77]], [
    [41, 88], [37, 91], [37, 84], [43, 97], [41, 77], [75, 108], [73, 64], [111, 115], [115, 58], [143, 111], [143, 58], [141, 97], [139, 73], [185, 93], [187, 77], [229, 93], [229, 75]], [[87, 115], [85, 121], [85, 113], [87, 132], [87, 113], [113, 139], [111, 122], [153, 126], [139, 121], [153, 99], [171, 139], [167, 115], [163, 106], [149, 60], [173, 60], [183, 106], [223, 71]], [[205, 38], [205, 40], [203, 38], [197, 56], [193, 42], [177, 62], [177, 53], [199, 33], [195, 25], [205, 47], [225, 14], [137, 99], [139, 80], [159, 152], [157, 152], [205, 144], [205, 144]], [[33, 84], [33, 84], [29, 84], [33, 66], [31, 86], [59, 53], [59, 102], [101, 49], [101, 108], [133, 53], [131, 108], [133, 66], [133, 91], [187, 67], [185, 91], [241, 66], [235, 95]]]

# TODO: send real videos
VIDEOS = os.listdir('webapp/static/videos/out/')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# create an instance for the ml queue for handling the inference at the backend
mlq = MLQ(q_name='pose_transfer', redis_host='localhost', redis_port=6379, redis_db=0)


# Routes for MLQ (GPU server)
@app.route('/do_transfer', methods=['POST'])
def do_pose_transfer():
    # request the backend to transfer the incoming image
    # but for now this is just a dummy for testing mlq
    assert request.json['number']
    job_id = mlq.post(request.json, CALLBACK_URL)
    return jsonify({'msg': 'Processing, check back soon.', 'job_id': job_id})

@app.route('/api/status/<job_id>', methods=['GET'])
def get_progress(job_id):
    return mlq.get_progress(job_id=job_id)

@app.route('/api/result/<job_id>', methods=['GET'])
def get_result(job_id):
    job = mlq.get_job(job_id=job_id)
    return job['result'] or '[not available yet]'

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

        # TODO: check for the target pose
        if 'inputPose' not in request.form or not request.form['inputPose']:
            flash('Please select a target pose')
            return redirect(request.url)
        else:
            try:
                pose_id = int(request.form['inputPose'])
                target_pose = POSES[pose_id]
            except ValueError:
                flash('Pose ID is invalid')
                return redirect(request.url)

        # TODO: make the form data serializable
        source_image_file = io.BytesIO()
        source_image.save(source_image_file, format="PNG")
        source_image = base64.b64encode(source_image_file.getvalue()).decode()
        payload = {
            'source_image': source_image,
            'target_pose': target_pose,
            'is_video': False}
        
        # send the posted data to the worker
        job_id = mlq.post(payload)
        return job_id
    elif request.method == 'GET':
        return render_template('index.html', poses=POSES)
    else:
        return f'error {request.method} method not implemented'

# app routes for serving the website
@app.route('/videos', methods=['GET', 'POST'])
def videos():
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

        # TODO: check for the target pose
        if 'inputPose' not in request.form or not request.form['inputPose']:
            flash('Please select a target pose')
            return redirect(request.url)
        else:
            try:
                pose_id = int(request.form['inputPose'])
                # TODO: send the right video data to the backend
                target_pose = VIDEOS[pose_id].split('.')[0]
                print('Here is the target_pose: ', target_pose)
            except ValueError:
                flash('Pose ID is invalid')
                return redirect(request.url)

        source_image_file = io.BytesIO()
        source_image.save(source_image_file, format="PNG")
        source_image = base64.b64encode(source_image_file.getvalue()).decode()
        payload = {
            'source_image': source_image,
            'target_pose': target_pose,
            'is_video': True}
        
        # send the posted data to the worker
        job_id = mlq.post(payload)
        return job_id
    elif request.method == 'GET':
        return render_template('videos.html', videos=VIDEOS)
    else:
        return f'error {request.method} method not implemented'

@app.route('/impressum')
def impressum():
    return render_template('impressum.html')

with app.test_request_context():
    print(url_for('index'))
    print(url_for('impressum'))

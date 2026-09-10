"""
HTTP API for the body-measurement pipeline.

Place in the root of Human-Body-Measurements-using-Computer-Vision/ and run
from there (the repo hardcodes relative paths to models/, data/, src/).

    pip install flask
    python measure_server.py --port 8080

Endpoints
---------
GET    /health                 model status
POST   /measure                multipart: image=<file>, height=<float>,
                               units=<in|cm>  ->  JSON incl. "id"
GET    /result/<id>            the JSON again
GET    /mesh/<id>              download the .obj for that id
GET    /results                list stored ids
DELETE /result/<id>            delete an id and its files

Example
-------
    curl -F image=@photo.jpg -F height=68 http://localhost:8080/measure
    curl -O -J http://localhost:8080/mesh/<id>
"""

from __future__ import absolute_import, division, print_function

import argparse
import datetime
import json
import os
import re
import shutil
import threading
import traceback
import uuid

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, jsonify, request
from PIL import Image
from six.moves import urllib

import extract_measurements
import utils
from src.RunModel import RunModel
from src.util import image as img_util

try:
    _RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    _RESAMPLE = Image.ANTIALIAS

PERSON_CLASS = 15
STORAGE_DIR = 'api_storage'
ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ID_RE = re.compile(r'^[0-9a-f]{32}$')

DEEPLAB_DIR = 'deeplab_model'
DEEPLAB_TARBALL = 'deeplabv3_pascal_trainval_2018_01_04.tar.gz'
DEEPLAB_URL = 'http://download.tensorflow.org/models/' + DEEPLAB_TARBALL

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES

# TF 1.x graphs are not safe under concurrent use; serialize all inference.
_INFER_LOCK = threading.Lock()
_STATE = {'ready': False, 'segmenter': None, 'hmr': None,
          'cp': None, 'faces': None}


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------

class DeepLabModel(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        import tarfile
        self.graph = tf.Graph()
        graph_def = None
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                graph_def = tf.GraphDef.FromString(
                    tar_file.extractfile(tar_info).read())
                break
        tar_file.close()
        if graph_def is None:
            raise RuntimeError('No frozen graph in %s' % tarball_path)
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        width, height = image.size
        ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        resized = image.convert('RGB').resize(
            (int(ratio * width), int(ratio * height)), _RESAMPLE)
        return self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized)]})[0]


def ensure_deeplab():
    if not os.path.exists(DEEPLAB_DIR):
        os.makedirs(DEEPLAB_DIR)
    path = os.path.join(DEEPLAB_DIR, DEEPLAB_TARBALL)
    if not os.path.exists(path):
        app.logger.info('Downloading DeepLab (~440 MB), one time only...')
        urllib.request.urlretrieve(DEEPLAB_URL, path)
    return path


def load_models():
    for required in ('models/model.ckpt-667589.index',
                     'models/neutral_smpl_with_cocoplus_reg.pkl',
                     'data/customBodyPoints.txt',
                     'src/tf_smpl/smpl_faces.npy'):
        if not os.path.exists(required):
            raise SystemExit(
                'Missing %s -- run this from the repo root and make sure the '
                'pretrained models and customBodyPoints.txt are downloaded.'
                % required)

    _STATE['segmenter'] = DeepLabModel(ensure_deeplab())
    app.logger.info('DeepLab ready.')
    _STATE['hmr'] = RunModel(sess=tf.Session())
    app.logger.info('HMR ready.')
    _STATE['cp'] = extract_measurements.convert_cp()
    _STATE['faces'] = np.load('./src/tf_smpl/smpl_faces.npy')
    _STATE['ready'] = True


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

def remove_background(pil_image):
    seg = _STATE['segmenter'].run(pil_image)
    seg = cv2.resize(seg.astype(np.uint8), pil_image.size,
                     interpolation=cv2.INTER_NEAREST)
    mask = 255 * (seg == PERSON_CLASS).astype(np.uint8)
    coverage = float((mask > 0).sum()) / mask.size
    img = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    subject = cv2.bitwise_and(img, img, mask=mask)
    return subject + (255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)), coverage


def preprocess(img_bgr, img_size=224):
    img = img_bgr[:, :, :3] if img_bgr.shape[2] == 4 else img_bgr
    scale = 1.0 if np.max(img.shape[:2]) == img_size \
        else float(img_size) / np.max(img.shape[:2])
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)[::-1]
    crop, _ = img_util.scale_and_crop(img, scale, center, img_size)
    return 2 * ((crop / 255.) - 0.5)


def write_obj(path, verts, faces):
    with open(path, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces:
            fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))


def run_pipeline(pil_image, height):
    """Returns (measurements dict, betas list, coverage, verts)."""
    with _INFER_LOCK:
        bg_removed, coverage = remove_background(pil_image)
        crop = preprocess(bg_removed)
        _, verts, _, _, theta = _STATE['hmr'].predict(
            np.expand_dims(crop, 0), get_theta=True)
    values = extract_measurements.calc_measure(
        _STATE['cp'], verts[0], height).flatten()
    measurements = dict(
        (label, round(float(v), 2)) for label, v in zip(utils.M_STR, values))
    betas = [round(float(b), 5) for b in theta[0][3 + 72:]]
    return measurements, betas, coverage, verts[0]


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------

def record_dir(record_id):
    if not ID_RE.match(record_id or ''):
        return None
    path = os.path.join(STORAGE_DIR, record_id)
    return path if os.path.isdir(path) else None


def load_record(record_id):
    d = record_dir(record_id)
    if d is None:
        return None
    try:
        with open(os.path.join(d, 'result.json')) as fh:
            return json.load(fh)
    except (IOError, ValueError):
        return None


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok' if _STATE['ready'] else 'loading',
        'measurements': list(utils.M_STR),
        'stored': len(os.listdir(STORAGE_DIR)) if os.path.isdir(STORAGE_DIR) else 0,
    })


@app.route('/measure', methods=['POST'])
def measure():
    if not _STATE['ready']:
        return jsonify({'error': 'models still loading'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'no file under form field "image"'}), 400
    upload = request.files['image']
    if not upload.filename:
        return jsonify({'error': 'empty filename'}), 400

    ext = os.path.splitext(upload.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({'error': 'unsupported extension %s' % ext,
                        'allowed': sorted(ALLOWED_EXTS)}), 400

    raw_height = request.form.get('height')
    if raw_height is None:
        return jsonify({'error': 'height is required; all measurements are '
                                 'scaled by it'}), 400
    try:
        height = float(raw_height)
    except ValueError:
        return jsonify({'error': 'height must be a number'}), 400
    if not 20.0 <= height <= 250.0:
        return jsonify({'error': 'height out of range (20-250)'}), 400

    units = request.form.get('units', 'in')
    if units not in ('in', 'cm'):
        return jsonify({'error': 'units must be "in" or "cm"'}), 400

    record_id = uuid.uuid4().hex
    out_dir = os.path.join(STORAGE_DIR, record_id)
    os.makedirs(out_dir)

    try:
        image_path = os.path.join(out_dir, 'input' + ext)
        upload.save(image_path)
        pil = Image.open(image_path)

        measurements, betas, coverage, verts = run_pipeline(pil, height)
        write_obj(os.path.join(out_dir, 'mesh.obj'), verts, _STATE['faces'])

        warnings = []
        if coverage < 0.03:
            warnings.append('person occupies only %.1f%% of the frame; '
                            'measurements are unreliable' % (100 * coverage))
        if coverage > 0.75:
            warnings.append('person fills the frame; body may be cropped')

        result = {
            'id': record_id,
            'created_at': datetime.datetime.utcnow().isoformat() + 'Z',
            'height_input': height,
            'units': units,
            'mask_coverage': round(coverage, 4),
            'measurements': measurements,
            'betas': betas,
            'mesh_url': '/mesh/%s' % record_id,
            'warnings': warnings,
        }
        with open(os.path.join(out_dir, 'result.json'), 'w') as fh:
            json.dump(result, fh, indent=2)
        return jsonify(result), 201

    except Exception as exc:
        shutil.rmtree(out_dir, ignore_errors=True)
        app.logger.error(traceback.format_exc())
        return jsonify({'error': 'inference failed',
                        'detail': '%s: %s' % (type(exc).__name__, exc)}), 500


@app.route('/result/<record_id>', methods=['GET'])
def get_result(record_id):
    result = load_record(record_id)
    if result is None:
        return jsonify({'error': 'unknown id'}), 404
    return jsonify(result)


@app.route('/mesh/<record_id>', methods=['GET'])
def get_mesh(record_id):
    d = record_dir(record_id)
    if d is None:
        return jsonify({'error': 'unknown id'}), 404
    mesh_path = os.path.join(d, 'mesh.obj')
    if not os.path.exists(mesh_path):
        return jsonify({'error': 'no mesh for this id'}), 404
    with open(mesh_path, 'rb') as fh:
        payload = fh.read()
    return Response(payload, mimetype='model/obj', headers={
        'Content-Disposition': 'attachment; filename="%s.obj"' % record_id,
        'Content-Length': str(len(payload)),
    })


@app.route('/results', methods=['GET'])
def list_results():
    if not os.path.isdir(STORAGE_DIR):
        return jsonify({'count': 0, 'results': []})
    items = []
    for record_id in os.listdir(STORAGE_DIR):
        result = load_record(record_id)
        if result:
            items.append({'id': record_id,
                          'created_at': result.get('created_at'),
                          'height_input': result.get('height_input'),
                          'units': result.get('units')})
    items.sort(key=lambda r: r.get('created_at') or '', reverse=True)
    return jsonify({'count': len(items), 'results': items})


@app.route('/result/<record_id>', methods=['DELETE'])
def delete_result(record_id):
    d = record_dir(record_id)
    if d is None:
        return jsonify({'error': 'unknown id'}), 404
    shutil.rmtree(d, ignore_errors=True)
    return jsonify({'deleted': record_id})


@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': 'file too large',
                    'max_bytes': MAX_UPLOAD_BYTES}), 413


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--storage', default=STORAGE_DIR)
    args = ap.parse_args()

    global STORAGE_DIR
    STORAGE_DIR = args.storage
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)

    load_models()
    # threaded so /mesh and /result stay responsive; inference is lock-guarded
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == '__main__':
    main()

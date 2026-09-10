"""
Batch body-measurement inference.

Drop this in the root of Human-Body-Measurements-using-Computer-Vision/
and run it from there (the repo hardcodes relative paths to models/ and data/).

Loads DeepLab + HMR once, then loops over a folder of images.
Does NOT import demo.py or inference.py, so it works even while demo.py
is broken at HEAD.

Usage:
    python batch_measure.py -i my_photos/ -ht 68
    python batch_measure.py -i my_photos/ --heights heights.csv --save-obj
    python batch_measure.py -i my_photos/ -ht 68 --units in -o results.csv

heights.csv format (one row per image, header required):
    filename,height
    alice.jpg,64.5
    bob.jpg,71
"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import os
import tarfile
import traceback

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from six.moves import urllib

import extract_measurements
import utils
from src.RunModel import RunModel
from src.util import image as img_util

# Pillow 10 removed Image.ANTIALIAS
try:
    _RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    _RESAMPLE = Image.ANTIALIAS

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
PERSON_CLASS = 15  # PASCAL VOC label index for "person"

DEEPLAB_DIR = 'deeplab_model'
DEEPLAB_TARBALL = 'deeplabv3_pascal_trainval_2018_01_04.tar.gz'
DEEPLAB_URL = 'http://download.tensorflow.org/models/' + DEEPLAB_TARBALL


class DeepLabModel(object):
    """Loads the frozen DeepLab v3 graph and runs person segmentation."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
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
            raise RuntimeError('No frozen inference graph in %s' % tarball_path)
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        width, height = image.size
        ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target = (int(ratio * width), int(ratio * height))
        resized = image.convert('RGB').resize(target, _RESAMPLE)
        seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized)]})[0]
        return resized, seg_map


def ensure_deeplab():
    if not os.path.exists(DEEPLAB_DIR):
        os.makedirs(DEEPLAB_DIR)
    path = os.path.join(DEEPLAB_DIR, DEEPLAB_TARBALL)
    if not os.path.exists(path):
        print('Downloading DeepLab (~440 MB), this happens once...')
        urllib.request.urlretrieve(DEEPLAB_URL, path)
        print('Done.')
    return path


def remove_background(pil_image, segmenter):
    """White out everything that isn't the person. Returns BGR uint8."""
    _, seg = segmenter.run(pil_image)
    seg = cv2.resize(seg.astype(np.uint8), pil_image.size,
                     interpolation=cv2.INTER_NEAREST)
    mask = (255 * (seg == PERSON_CLASS).astype(np.uint8))

    coverage = float((mask > 0).sum()) / mask.size
    img = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    subject = cv2.bitwise_and(img, img, mask=mask)
    white_bg = 255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return subject + white_bg, coverage


def preprocess(img_bgr, img_size=224):
    """Center-crop + scale to img_size, normalize to [-1, 1]. From demo.py."""
    img = img_bgr
    if img.shape[2] == 4:
        img = img[:, :, :3]
    scale = 1.0 if np.max(img.shape[:2]) == img_size \
        else float(img_size) / np.max(img.shape[:2])
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)[::-1]  # (x, y)
    crop, proc_param = img_util.scale_and_crop(img, scale, center, img_size)
    crop = 2 * ((crop / 255.) - 0.5)
    return crop, proc_param


def save_obj(path, verts, faces):
    with open(path, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces:
            fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))


def load_heights(csv_path):
    heights = {}
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            heights[row['filename'].strip()] = float(row['height'])
    return heights


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-i', '--input', required=True,
                    help='Image file or directory of images')
    ap.add_argument('-ht', '--height', type=float, default=None,
                    help='Subject height, applied to all images')
    ap.add_argument('--heights', default=None,
                    help='CSV with per-image heights (filename,height)')
    ap.add_argument('-o', '--output', default='measurements.csv')
    ap.add_argument('--units', default='in',
                    help='Label only; outputs match your height unit')
    ap.add_argument('--save-obj', action='store_true',
                    help='Write a .obj mesh per subject into obj_out/')
    ap.add_argument('--min-coverage', type=float, default=0.03,
                    help='Skip images where the person mask is smaller than '
                         'this fraction of the frame (default 0.03)')
    args = ap.parse_args()

    if args.height is None and args.heights is None:
        ap.error('Give either -ht or --heights; scale comes entirely from height.')

    per_image_heights = load_heights(args.heights) if args.heights else {}

    if os.path.isdir(args.input):
        files = sorted(f for f in os.listdir(args.input)
                       if f.lower().endswith(IMG_EXTS))
        files = [os.path.join(args.input, f) for f in files]
    else:
        files = [args.input]
    if not files:
        raise SystemExit('No images found in %s' % args.input)
    print('Found %d image(s).' % len(files))

    # --- Load both models once ---
    segmenter = DeepLabModel(ensure_deeplab())
    print('DeepLab ready.')
    sess = tf.Session()
    hmr = RunModel(sess=sess)
    print('HMR ready.')

    control_points = extract_measurements.convert_cp()
    faces = np.load('./src/tf_smpl/smpl_faces.npy') if args.save_obj else None
    if args.save_obj and not os.path.exists('obj_out'):
        os.makedirs('obj_out')

    rows, skipped = [], []
    for path in files:
        name = os.path.basename(path)
        height = per_image_heights.get(name, args.height)
        if height is None:
            skipped.append((name, 'no height given'))
            continue

        try:
            pil = Image.open(path)
            bg_removed, coverage = remove_background(pil, segmenter)
            if coverage < args.min_coverage:
                skipped.append((name, 'person mask only %.1f%% of frame'
                                % (100 * coverage)))
                continue

            crop, _ = preprocess(bg_removed)
            _, verts, _, _, theta = hmr.predict(
                np.expand_dims(crop, 0), get_theta=True)

            measures = extract_measurements.calc_measure(
                control_points, verts[0], height).flatten()
            betas = theta[0][3 + 72:]  # 10 SMPL shape coefficients

            row = {'filename': name, 'height_input': height,
                   'mask_coverage': round(coverage, 4)}
            for label, value in zip(utils.M_STR, measures):
                row[label] = round(float(value), 2)
            for j, b in enumerate(betas):
                row['beta_%d' % j] = round(float(b), 5)
            rows.append(row)

            if args.save_obj:
                save_obj(os.path.join('obj_out', name + '.obj'),
                         verts[0], faces)

            print('  %-28s ok  (mask %.1f%%)' % (name, 100 * coverage))
        except Exception as exc:
            skipped.append((name, '%s: %s' % (type(exc).__name__, exc)))
            traceback.print_exc()

    if rows:
        fields = (['filename', 'height_input', 'mask_coverage']
                  + list(utils.M_STR)
                  + ['beta_%d' % j for j in range(10)])
        with open(args.output, 'w') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print('\nWrote %d row(s) to %s (units: %s)'
              % (len(rows), args.output, args.units))

    if skipped:
        print('\nSkipped %d image(s):' % len(skipped))
        for name, why in skipped:
            print('  %-28s %s' % (name, why))


if __name__ == '__main__':
    main()

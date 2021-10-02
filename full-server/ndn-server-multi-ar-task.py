import numpy as np

import NDNstreaming1
import NDNstreaming2
import threading
import time
import cv2
import ffmpeg
from ndn.app import NDNApp
from ndn.encoding import Name, InterestParam, BinaryStr, FormalName, MetaInfo, Component
from ndn.security import DigestSha256Signer, Sha256WithEcdsaSigner, Sha256WithRsaSigner, HmacSha256Signer
import logging
import time
from collections import deque

# Libraries for yolo
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import torch
import torch.backends.cudnn as cudnn
import argparse
from typing import Optional
import sys,os
import trimesh
from pyrender import PerspectiveCamera, SpotLight, Mesh, Node, Scene, OffscreenRenderer, RenderFlags

from facenet_pytorch import MTCNN
import paddlehub as hub
from PIL import Image
from Crypto.Cipher import AES


try:
    sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

device_type = "gpu"  # "gpu" or "cpu"
height = 1080
width = 1920
logging.basicConfig(level=logging.DEBUG)

try:
    import thread
except ImportError:
    import _thread as thread

dth = threading.Thread(target=NDNstreaming1.run, args=('testecho', width, height, "", device_type))
# dth2 = threading.Thread(target=NDNstreaming2.run, args=('testecho2',))
dth.start()
# dth2.start()

crf = 30
interval = 1. / crf
if device_type == "gpu":
    encoder = (
        ffmpeg
        .input("pipe:", format='rawvideo', s='{}x{}'.format(width, height), pix_fmt='bgr24')
        # .input('testsrc=size={}x{}:rate=30:duration=100'.format(width, height), f='lavfi')
        # .output("pipe:", format='h264', vcodec='libx264', crf=crf, g=crf, keyint_min=crf, bf=0, bitrate='320k',pix_fmt='yuv420p') # previous ffmpeg
        # .output("pipe:", format='h264', vcodec='h264_nvenc', crf=crf, g=crf, keyint_min=crf, bf=0, bitrate='320k', pix_fmt='yuv420p')
        .output("pipe:", format='h264', vcodec='h264_nvenc', crf=crf, g=crf, keyint_min=crf, bf=0, bitrate='320k',
                profile="baseline", pix_fmt='yuv420p')  # 720p
        # .output("pipe:", format='h264', vcodec='h264_nvenc', crf=crf, g=crf, keyint_min=crf, bf=0, bitrate='2000k',profile="baseline", pix_fmt='yuv420p') # 1080p
        # .overwrite_output()
        .video
        .run_async(pipe_stdin=True, pipe_stdout=True)
    )
else:
    encoder = (
        ffmpeg
            .input("pipe:", format='rawvideo', s='{}x{}'.format(width, height), pix_fmt='bgr24')
            # .input('testsrc=size={}x{}:rate=30:duration=100'.format(width, height), f='lavfi')
            .output("pipe:", format='h264', vcodec='libx264', crf=crf, g=crf, keyint_min=crf, bf=0,
                    pix_fmt='yuv420p')
            # .overwrite_output()
            .video
            .run_async(pipe_stdin=True, pipe_stdout=True)
    )

producer_frame = np.zeros([height, width, 3])
# right_frame = np.zeros([height, width, 3])
# display_image = np.hstack((left_frame, right_frame))
display_image = producer_frame
app = NDNApp()

buffer_time = 100
current_I_frame = 1
I_frame_index = 0
frame_buffer_I = deque(maxlen=buffer_time)
frame_buffer_P = deque(maxlen=buffer_time * crf)

frame_buffer = deque(maxlen=buffer_time)
frame_buffer_dict = {}
interest_buffer = deque(maxlen=buffer_time)


# Sign the data packet with the edge key /edge/KEY/000-+
private_key = bytes([
    0x30, 0x82, 0x04, 0xbf, 0x02, 0x01, 0x00, 0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7,
    0x0d, 0x01, 0x01, 0x01, 0x05, 0x00, 0x04, 0x82, 0x04, 0xa9, 0x30, 0x82, 0x04, 0xa5, 0x02, 0x01,
    0x00, 0x02, 0x82, 0x01, 0x01, 0x00, 0xb8, 0x09, 0xa7, 0x59, 0x82, 0x84, 0xec, 0x4f, 0x06, 0xfa,
    0x1c, 0xb2, 0xe1, 0x38, 0x93, 0x53, 0xbb, 0x7d, 0xd4, 0xac, 0x88, 0x1a, 0xf8, 0x25, 0x11, 0xe4,
    0xfa, 0x1d, 0x61, 0x24, 0x5b, 0x82, 0xca, 0xcd, 0x72, 0xce, 0xdb, 0x66, 0xb5, 0x8d, 0x54, 0xbd,
    0xfb, 0x23, 0xfd, 0xe8, 0x8e, 0xaf, 0xa7, 0xb3, 0x79, 0xbe, 0x94, 0xb5, 0xb7, 0xba, 0x17, 0xb6,
    0x05, 0xae, 0xce, 0x43, 0xbe, 0x3b, 0xce, 0x6e, 0xea, 0x07, 0xdb, 0xbf, 0x0a, 0x7e, 0xeb, 0xbc,
    0xc9, 0x7b, 0x62, 0x3c, 0xf5, 0xe1, 0xce, 0xe1, 0xd9, 0x8d, 0x9c, 0xfe, 0x1f, 0xc7, 0xf8, 0xfb,
    0x59, 0xc0, 0x94, 0x0b, 0x2c, 0xd9, 0x7d, 0xbc, 0x96, 0xeb, 0xb8, 0x79, 0x22, 0x8a, 0x2e, 0xa0,
    0x12, 0x1d, 0x42, 0x07, 0xb6, 0x5d, 0xdb, 0xe1, 0xf6, 0xb1, 0x5d, 0x7b, 0x1f, 0x54, 0x52, 0x1c,
    0xa3, 0x11, 0x9b, 0xf9, 0xeb, 0xbe, 0xb3, 0x95, 0xca, 0xa5, 0x87, 0x3f, 0x31, 0x18, 0x1a, 0xc9,
    0x99, 0x01, 0xec, 0xaa, 0x90, 0xfd, 0x8a, 0x36, 0x35, 0x5e, 0x12, 0x81, 0xbe, 0x84, 0x88, 0xa1,
    0x0d, 0x19, 0x2a, 0x4a, 0x66, 0xc1, 0x59, 0x3c, 0x41, 0x83, 0x3d, 0x3d, 0xb8, 0xd4, 0xab, 0x34,
    0x90, 0x06, 0x3e, 0x1a, 0x61, 0x74, 0xbe, 0x04, 0xf5, 0x7a, 0x69, 0x1b, 0x9d, 0x56, 0xfc, 0x83,
    0xb7, 0x60, 0xc1, 0x5e, 0x9d, 0x85, 0x34, 0xfd, 0x02, 0x1a, 0xba, 0x2c, 0x09, 0x72, 0xa7, 0x4a,
    0x5e, 0x18, 0xbf, 0xc0, 0x58, 0xa7, 0x49, 0x34, 0x46, 0x61, 0x59, 0x0e, 0xe2, 0x6e, 0x9e, 0xd2,
    0xdb, 0xfd, 0x72, 0x2f, 0x3c, 0x47, 0xcc, 0x5f, 0x99, 0x62, 0xee, 0x0d, 0xf3, 0x1f, 0x30, 0x25,
    0x20, 0x92, 0x15, 0x4b, 0x04, 0xfe, 0x15, 0x19, 0x1d, 0xdc, 0x7e, 0x5c, 0x10, 0x21, 0x52, 0x21,
    0x91, 0x54, 0x60, 0x8b, 0x92, 0x41, 0x02, 0x03, 0x01, 0x00, 0x01, 0x02, 0x82, 0x01, 0x01, 0x00,
    0x8a, 0x05, 0xfb, 0x73, 0x7f, 0x16, 0xaf, 0x9f, 0xa9, 0x4c, 0xe5, 0x3f, 0x26, 0xf8, 0x66, 0x4d,
    0xd2, 0xfc, 0xd1, 0x06, 0xc0, 0x60, 0xf1, 0x9f, 0xe3, 0xa6, 0xc6, 0x0a, 0x48, 0xb3, 0x9a, 0xca,
    0x21, 0xcd, 0x29, 0x80, 0x88, 0x3d, 0xa4, 0x85, 0xa5, 0x7b, 0x82, 0x21, 0x81, 0x28, 0xeb, 0xf2,
    0x43, 0x24, 0xb0, 0x76, 0xc5, 0x52, 0xef, 0xc2, 0xea, 0x4b, 0x82, 0x41, 0x92, 0xc2, 0x6d, 0xa6,
    0xae, 0xf0, 0xb2, 0x26, 0x48, 0xa1, 0x23, 0x7f, 0x02, 0xcf, 0xa8, 0x90, 0x17, 0xa2, 0x3e, 0x8a,
    0x26, 0xbd, 0x6d, 0x8a, 0xee, 0xa6, 0x0c, 0x31, 0xce, 0xc2, 0xbb, 0x92, 0x59, 0xb5, 0x73, 0xe2,
    0x7d, 0x91, 0x75, 0xe2, 0xbd, 0x8c, 0x63, 0xe2, 0x1c, 0x8b, 0xc2, 0x6a, 0x1c, 0xfe, 0x69, 0xc0,
    0x44, 0xcb, 0x58, 0x57, 0xb7, 0x13, 0x42, 0xf0, 0xdb, 0x50, 0x4c, 0xe0, 0x45, 0x09, 0x8f, 0xca,
    0x45, 0x8a, 0x06, 0xfe, 0x98, 0xd1, 0x22, 0xf5, 0x5a, 0x9a, 0xdf, 0x89, 0x17, 0xca, 0x20, 0xcc,
    0x12, 0xa9, 0x09, 0x3d, 0xd5, 0xf7, 0xe3, 0xeb, 0x08, 0x4a, 0xc4, 0x12, 0xc0, 0xb9, 0x47, 0x6c,
    0x79, 0x50, 0x66, 0xa3, 0xf8, 0xaf, 0x2c, 0xfa, 0xb4, 0x6b, 0xec, 0x03, 0xad, 0xcb, 0xda, 0x24,
    0x0c, 0x52, 0x07, 0x87, 0x88, 0xc0, 0x21, 0xf3, 0x02, 0xe8, 0x24, 0x44, 0x0f, 0xcd, 0xa0, 0xad,
    0x2f, 0x1b, 0x79, 0xab, 0x6b, 0x49, 0x4a, 0xe6, 0x3b, 0xd0, 0xad, 0xc3, 0x48, 0xb9, 0xf7, 0xf1,
    0x34, 0x09, 0xeb, 0x7a, 0xc0, 0xd5, 0x0d, 0x39, 0xd8, 0x45, 0xce, 0x36, 0x7a, 0xd8, 0xde, 0x3c,
    0xb0, 0x21, 0x96, 0x97, 0x8a, 0xff, 0x8b, 0x23, 0x60, 0x4f, 0xf0, 0x3d, 0xd7, 0x8f, 0xf3, 0x2c,
    0xcb, 0x1d, 0x48, 0x3f, 0x86, 0xc4, 0xa9, 0x00, 0xf2, 0x23, 0x2d, 0x72, 0x4d, 0x66, 0xa5, 0x01,
    0x02, 0x81, 0x81, 0x00, 0xdc, 0x4f, 0x99, 0x44, 0x0d, 0x7f, 0x59, 0x46, 0x1e, 0x8f, 0xe7, 0x2d,
    0x8d, 0xdd, 0x54, 0xc0, 0xf7, 0xfa, 0x46, 0x0d, 0x9d, 0x35, 0x03, 0xf1, 0x7c, 0x12, 0xf3, 0x5a,
    0x9d, 0x83, 0xcf, 0xdd, 0x37, 0x21, 0x7c, 0xb7, 0xee, 0xc3, 0x39, 0xd2, 0x75, 0x8f, 0xb2, 0x2d,
    0x6f, 0xec, 0xc6, 0x03, 0x55, 0xd7, 0x00, 0x67, 0xd3, 0x9b, 0xa2, 0x68, 0x50, 0x6f, 0x9e, 0x28,
    0xa4, 0x76, 0x39, 0x2b, 0xb2, 0x65, 0xcc, 0x72, 0x82, 0x93, 0xa0, 0xcf, 0x10, 0x05, 0x6a, 0x75,
    0xca, 0x85, 0x35, 0x99, 0xb0, 0xa6, 0xc6, 0xef, 0x4c, 0x4d, 0x99, 0x7d, 0x2c, 0x38, 0x01, 0x21,
    0xb5, 0x31, 0xac, 0x80, 0x54, 0xc4, 0x18, 0x4b, 0xfd, 0xef, 0xb3, 0x30, 0x22, 0x51, 0x5a, 0xea,
    0x7d, 0x9b, 0xb2, 0x9d, 0xcb, 0xba, 0x3f, 0xc0, 0x1a, 0x6b, 0xcd, 0xb0, 0xe6, 0x2f, 0x04, 0x33,
    0xd7, 0x3a, 0x49, 0x71, 0x02, 0x81, 0x81, 0x00, 0xd5, 0xd9, 0xc9, 0x70, 0x1a, 0x13, 0xb3, 0x39,
    0x24, 0x02, 0xee, 0xb0, 0xbb, 0x84, 0x17, 0x12, 0xc6, 0xbd, 0x65, 0x73, 0xe9, 0x34, 0x5d, 0x43,
    0xff, 0xdc, 0xf8, 0x55, 0xaf, 0x2a, 0xb9, 0xe1, 0xfa, 0x71, 0x65, 0x4e, 0x50, 0x0f, 0xa4, 0x3b,
    0xe5, 0x68, 0xf2, 0x49, 0x71, 0xaf, 0x15, 0x88, 0xd7, 0xaf, 0xc4, 0x9d, 0x94, 0x84, 0x6b, 0x5b,
    0x10, 0xd5, 0xc0, 0xaa, 0x0c, 0x13, 0x62, 0x99, 0xc0, 0x8b, 0xfc, 0x90, 0x0f, 0x87, 0x40, 0x4d,
    0x58, 0x88, 0xbd, 0xe2, 0xba, 0x3e, 0x7e, 0x2d, 0xd7, 0x69, 0xa9, 0x3c, 0x09, 0x64, 0x31, 0xb6,
    0xcc, 0x4d, 0x1f, 0x23, 0xb6, 0x9e, 0x65, 0xd6, 0x81, 0xdc, 0x85, 0xcc, 0x1e, 0xf1, 0x0b, 0x84,
    0x38, 0xab, 0x93, 0x5f, 0x9f, 0x92, 0x4e, 0x93, 0x46, 0x95, 0x6b, 0x3e, 0xb6, 0xc3, 0x1b, 0xd7,
    0x69, 0xa1, 0x0a, 0x97, 0x37, 0x78, 0xed, 0xd1, 0x02, 0x81, 0x80, 0x33, 0x18, 0xc3, 0x13, 0x65,
    0x8e, 0x03, 0xc6, 0x9f, 0x90, 0x00, 0xae, 0x30, 0x19, 0x05, 0x6f, 0x3c, 0x14, 0x6f, 0xea, 0xf8,
    0x6b, 0x33, 0x5e, 0xee, 0xc7, 0xf6, 0x69, 0x2d, 0xdf, 0x44, 0x76, 0xaa, 0x32, 0xba, 0x1a, 0x6e,
    0xe6, 0x18, 0xa3, 0x17, 0x61, 0x1c, 0x92, 0x2d, 0x43, 0x5d, 0x29, 0xa8, 0xdf, 0x14, 0xd8, 0xff,
    0xdb, 0x38, 0xef, 0xb8, 0xb8, 0x2a, 0x96, 0x82, 0x8e, 0x68, 0xf4, 0x19, 0x8c, 0x42, 0xbe, 0xcc,
    0x4a, 0x31, 0x21, 0xd5, 0x35, 0x6c, 0x5b, 0xa5, 0x7c, 0xff, 0xd1, 0x85, 0x87, 0x28, 0xdc, 0x97,
    0x75, 0xe8, 0x03, 0x80, 0x1d, 0xfd, 0x25, 0x34, 0x41, 0x31, 0x21, 0x12, 0x87, 0xe8, 0x9a, 0xb7,
    0x6a, 0xc0, 0xc4, 0x89, 0x31, 0x15, 0x45, 0x0d, 0x9c, 0xee, 0xf0, 0x6a, 0x2f, 0xe8, 0x59, 0x45,
    0xc7, 0x7b, 0x0d, 0x6c, 0x55, 0xbb, 0x43, 0xca, 0xc7, 0x5a, 0x01, 0x02, 0x81, 0x81, 0x00, 0xab,
    0xf4, 0xd5, 0xcf, 0x78, 0x88, 0x82, 0xc2, 0xdd, 0xbc, 0x25, 0xe6, 0xa2, 0xc1, 0xd2, 0x33, 0xdc,
    0xef, 0x0a, 0x97, 0x2b, 0xdc, 0x59, 0x6a, 0x86, 0x61, 0x4e, 0xa6, 0xc7, 0x95, 0x99, 0xa6, 0xa6,
    0x55, 0x6c, 0x5a, 0x8e, 0x72, 0x25, 0x63, 0xac, 0x52, 0xb9, 0x10, 0x69, 0x83, 0x99, 0xd3, 0x51,
    0x6c, 0x1a, 0xb3, 0x83, 0x6a, 0xff, 0x50, 0x58, 0xb7, 0x28, 0x97, 0x13, 0xe2, 0xba, 0x94, 0x5b,
    0x89, 0xb4, 0xea, 0xba, 0x31, 0xcd, 0x78, 0xe4, 0x4a, 0x00, 0x36, 0x42, 0x00, 0x62, 0x41, 0xc6,
    0x47, 0x46, 0x37, 0xea, 0x6d, 0x50, 0xb4, 0x66, 0x8f, 0x55, 0x0c, 0xc8, 0x99, 0x91, 0xd5, 0xec,
    0xd2, 0x40, 0x1c, 0x24, 0x7d, 0x3a, 0xff, 0x74, 0xfa, 0x32, 0x24, 0xe0, 0x11, 0x2b, 0x71, 0xad,
    0x7e, 0x14, 0xa0, 0x77, 0x21, 0x68, 0x4f, 0xcc, 0xb6, 0x1b, 0xe8, 0x00, 0x49, 0x13, 0x21, 0x02,
    0x81, 0x81, 0x00, 0xb6, 0x18, 0x73, 0x59, 0x2c, 0x4f, 0x92, 0xac, 0xa2, 0x2e, 0x5f, 0xb6, 0xbe,
    0x78, 0x5d, 0x47, 0x71, 0x04, 0x92, 0xf0, 0xd7, 0xe8, 0xc5, 0x7a, 0x84, 0x6b, 0xb8, 0xb4, 0x30,
    0x1f, 0xd8, 0x0d, 0x58, 0xd0, 0x64, 0x80, 0xa7, 0x21, 0x1a, 0x48, 0x00, 0x37, 0xd6, 0x19, 0x71,
    0xbb, 0x91, 0x20, 0x9d, 0xe2, 0xc3, 0xec, 0xdb, 0x36, 0x1c, 0xca, 0x48, 0x7d, 0x03, 0x32, 0x74,
    0x1e, 0x65, 0x73, 0x02, 0x90, 0x73, 0xd8, 0x3f, 0xb5, 0x52, 0x35, 0x79, 0x1c, 0xee, 0x93, 0xa3,
    0x32, 0x8b, 0xed, 0x89, 0x98, 0xf1, 0x0c, 0xd8, 0x12, 0xf2, 0x89, 0x7f, 0x32, 0x23, 0xec, 0x67,
    0x66, 0x52, 0x83, 0x89, 0x99, 0x5e, 0x42, 0x2b, 0x42, 0x4b, 0x84, 0x50, 0x1b, 0x3e, 0x47, 0x6d,
    0x74, 0xfb, 0xd1, 0xa6, 0x10, 0x20, 0x6c, 0x6e, 0xbe, 0x44, 0x3f, 0xb9, 0xfe, 0xbc, 0x8d, 0xda,
    0xcb, 0xea, 0x8f
])
public_key = bytes([
    0x30, 0x82, 0x01, 0x22, 0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01,
    0x01, 0x05, 0x00, 0x03, 0x82, 0x01, 0x0f, 0x00, 0x30, 0x82, 0x01, 0x0a, 0x02, 0x82, 0x01, 0x01,
    0x00, 0xb8, 0x09, 0xa7, 0x59, 0x82, 0x84, 0xec, 0x4f, 0x06, 0xfa, 0x1c, 0xb2, 0xe1, 0x38, 0x93,
    0x53, 0xbb, 0x7d, 0xd4, 0xac, 0x88, 0x1a, 0xf8, 0x25, 0x11, 0xe4, 0xfa, 0x1d, 0x61, 0x24, 0x5b,
    0x82, 0xca, 0xcd, 0x72, 0xce, 0xdb, 0x66, 0xb5, 0x8d, 0x54, 0xbd, 0xfb, 0x23, 0xfd, 0xe8, 0x8e,
    0xaf, 0xa7, 0xb3, 0x79, 0xbe, 0x94, 0xb5, 0xb7, 0xba, 0x17, 0xb6, 0x05, 0xae, 0xce, 0x43, 0xbe,
    0x3b, 0xce, 0x6e, 0xea, 0x07, 0xdb, 0xbf, 0x0a, 0x7e, 0xeb, 0xbc, 0xc9, 0x7b, 0x62, 0x3c, 0xf5,
    0xe1, 0xce, 0xe1, 0xd9, 0x8d, 0x9c, 0xfe, 0x1f, 0xc7, 0xf8, 0xfb, 0x59, 0xc0, 0x94, 0x0b, 0x2c,
    0xd9, 0x7d, 0xbc, 0x96, 0xeb, 0xb8, 0x79, 0x22, 0x8a, 0x2e, 0xa0, 0x12, 0x1d, 0x42, 0x07, 0xb6,
    0x5d, 0xdb, 0xe1, 0xf6, 0xb1, 0x5d, 0x7b, 0x1f, 0x54, 0x52, 0x1c, 0xa3, 0x11, 0x9b, 0xf9, 0xeb,
    0xbe, 0xb3, 0x95, 0xca, 0xa5, 0x87, 0x3f, 0x31, 0x18, 0x1a, 0xc9, 0x99, 0x01, 0xec, 0xaa, 0x90,
    0xfd, 0x8a, 0x36, 0x35, 0x5e, 0x12, 0x81, 0xbe, 0x84, 0x88, 0xa1, 0x0d, 0x19, 0x2a, 0x4a, 0x66,
    0xc1, 0x59, 0x3c, 0x41, 0x83, 0x3d, 0x3d, 0xb8, 0xd4, 0xab, 0x34, 0x90, 0x06, 0x3e, 0x1a, 0x61,
    0x74, 0xbe, 0x04, 0xf5, 0x7a, 0x69, 0x1b, 0x9d, 0x56, 0xfc, 0x83, 0xb7, 0x60, 0xc1, 0x5e, 0x9d,
    0x85, 0x34, 0xfd, 0x02, 0x1a, 0xba, 0x2c, 0x09, 0x72, 0xa7, 0x4a, 0x5e, 0x18, 0xbf, 0xc0, 0x58,
    0xa7, 0x49, 0x34, 0x46, 0x61, 0x59, 0x0e, 0xe2, 0x6e, 0x9e, 0xd2, 0xdb, 0xfd, 0x72, 0x2f, 0x3c,
    0x47, 0xcc, 0x5f, 0x99, 0x62, 0xee, 0x0d, 0xf3, 0x1f, 0x30, 0x25, 0x20, 0x92, 0x15, 0x4b, 0x04,
    0xfe, 0x15, 0x19, 0x1d, 0xdc, 0x7e, 0x5c, 0x10, 0x21, 0x52, 0x21, 0x91, 0x54, 0x60, 0x8b, 0x92,
    0x41, 0x02, 0x03, 0x01, 0x00, 0x01
])
signer = Sha256WithRsaSigner('/edge/KEY/000', private_key)

# Encryption setting
cipher_key = bytearray(16)
enable_cipher = True


def encrypt_message(plaintext, key=cipher_key):
    if not enable_cipher:
        return plaintext
    iv = os.urandom(16)
    cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=128)
    return iv + cipher.encrypt(plaintext)


@app.route('/edge')
def on_interest(name: FormalName, param: InterestParam, _app_param: Optional[BinaryStr]):
    logging.info(f'>> I: {Name.to_str(name)}, {param}')
    request = Name.to_str(name).split("/")
    print("handle Interest Name", Name.to_str(name))
    if request[3] == "metadata":
        print("handle Meta data")
        # content = json.dumps(list(pred_frame_buffer)).encode()
        # content = str(current_I_frame).encode()
        content = Name.to_str(name + [Component.from_number(current_I_frame, 0)]).encode()
        name = name
        app.put_data(name, content=content, freshness_period=300)
        logging.info("handle to name " + Name.to_str(name))
    elif request[3] == "frame":
        # TODO: Sync IframeIndex in the Iframe
        interest_frame_num = int(request[-1])
        if interest_frame_num in frame_buffer_dict:
            content = frame_buffer_dict[interest_frame_num][0]
            app.put_data(name + [frame_buffer_dict[interest_frame_num][1]] + [b'\x08\x02\x00\x00'], content=content,
                         freshness_period=2000, final_block_id=Component.from_segment(0), signer=signer)
            print(f'handle interest: publish pending interest' + Name.to_str(name) + "------------/" + str(
                interest_frame_num) + "length: ", len(content))
        else:
            interest_buffer.append([interest_frame_num, name])
    elif request[2] == "KEY":
        content = public_key
        name = name
        app.put_data(name, content=content, freshness_period=3000)
    else:
        print("handle Request missing ", Name.to_str(name))

    while len(interest_buffer) > 0 and len(frame_buffer) > 0 and frame_buffer[-1] >= interest_buffer[0][0]:
        pendingInterest = interest_buffer.popleft()
        pendingFN = pendingInterest[0]
        pendingName = pendingInterest[1]
        if pendingFN in frame_buffer_dict:
            content = frame_buffer_dict[pendingFN][0]
            app.put_data(pendingName + [frame_buffer_dict[pendingFN][1]] + [b'\x08\x02\x00\x00'], content=content,
                         freshness_period=2000, final_block_id=Component.from_segment(0), signer=signer)
            print(f'handle interest: publish pending interest' + Name.to_str(pendingName) + "------------/" + str(
                pendingFN) + "length: ", len(content))


def resizeImg(im, desired_width, desired_height):
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_width) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_width - new_size[1]
    delta_h = desired_height - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    try:
        return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    except:
        print("handle resizeImg Error:", top, bottom, left, right, delta_h, delta_w, desired_height, new_size, old_size)
        return np.zeros(desired_height * desired_width * 3)


def overlay_transparent_image(bg, fg, x1, y1):
    # background is 3 RGB, foreground is 4 RGBA, x1 and y1 is the position of overlay
    bg = bg.copy()
    fg = cv2.resize(fg, (bg.shape[1], bg.shape[0]))
    # fg = fg.copy()
    h, w = fg.shape[:2]
    t = bg[y1:y1 + h, x1:x1 + w]
    b, g, r, a = cv2.split(fg)
    mask = cv2.merge((a, a, a))
    fg = cv2.merge((b, g, r))
    t = t.astype(float)
    fg = fg.astype(float)
    mask = mask.astype(float) / 255
    foreground = cv2.multiply(mask, fg)
    background = cv2.multiply(1.0 - mask, t)
    overlaid = cv2.add(foreground, background)
    bg[y1:y1 + h, x1:x1 + w] = overlaid
    return bg


def human_seg_light(human_seg, src_img):
    result = human_seg.segment(images=[src_img], visualization=False)[0]
    image_alpha = result['data'].astype(np.uint8)
    img_bg = Image.new('RGBA', (src_img.shape[1], src_img.shape[0]), (48, 255, 25, 255))
    image_temp = Image.fromarray(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGBA))
    img_bg.paste(image_temp, (0, 0), Image.fromarray(image_alpha))
    opencvImage = cv2.cvtColor(np.array(img_bg), cv2.COLOR_RGB2BGR)
    return opencvImage


def video_encoder():
    global left_frame, display_image
    # cv2.namedWindow("Output", cv2.WINDOW_GUI_NORMAL)
    # producer_frame = np.random.randint(0, 256, height * width * 3, dtype='uint8')

    task_type = args.task
    device = select_device(args.device)
    if task_type == "yolo":
        source, weights, view_img, save_txt, imgsz = args.source, args.weights, args.view_img, args.save_txt, args.img_size

        # Initialize
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # ======================================Load Yolo model=========================================================
        yoloModel = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(yoloModel.stride.max())
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            yoloModel.half()  # to FP16

        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Get names and colors
        yoloNames = yoloModel.module.names if hasattr(yoloModel, 'module') else yoloModel.names
        yoloColors = [[random.randint(0, 255) for _ in range(3)] for _ in yoloNames]

        if device.type != 'cpu':
            yoloModel(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(yoloModel.parameters())))  # run once
    elif task_type == "face":
        # ======================================Load Face Detection model=========================================================
        mtcnnModel = MTCNN(keep_all=True, device=device)
    elif task_type == "openpose":
        # ======================================Load Openpose model=========================================================
        try:
            params = dict()
            params["model_folder"] = "./models/"
            # params["face"] = True
            # params["hand"] = True
            # Starting OpenPose
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()
        except Exception as e:
            print(e)
            # sys.exit(-1)
    elif task_type == "modelrotate" or task_type == "armarker" or task_type == "arvideo":
        # ======================================Load AR Marker=========================================================
        markerTarget = cv2.imread('./marker/marker.jpg')
        markerH, markerW, markerC = markerTarget.shape
        orbModel = cv2.ORB_create(nfeatures=1000)
        marker_kp1, marker_des1 = orbModel.detectAndCompute(markerTarget, None)
        camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        last_matrix = np.ndarray(shape=(3, 3), dtype=float, order='F')

        # ======================================Load AR MP4=========================================================
        arVideo = cv2.VideoCapture('./marker/Tunnel.mp4')
        _, imgArVideo = arVideo.read()
        markerDetection = False
        arVideoFrameCounter = 0
        last_dst = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]]).reshape(-1, 1, 2)

        # ======================================Init AR 3D model=========================================================
        pyrenderScene = Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=(0, 0, 0, 0))
        # scene = Scene.from_trimesh_scene(bg_scene)
        fuze_trimesh = trimesh.load('./model3d/drill.obj')
        mesh = Mesh.from_trimesh(fuze_trimesh)
        mesh_node = Node(mesh=mesh, matrix=np.eye(4))
        pyrenderScene.add_node(mesh_node)
        # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
        camera = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
        s = np.sqrt(2) / 2
        camera_pose = np.array([
            [0.0, -s, s, 0.3],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, s, s, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ])
        # # add normal camera
        # scene.add(camera, pose=camera_pose)
        # Or add a camera node
        cam_node = Node(camera=camera, matrix=camera_pose)
        pyrenderScene.add_node(cam_node)
        light = SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi / 16.0)
        pyrenderScene.add(light, pose=camera_pose)
        offscreenRender = OffscreenRenderer(640, 480)
    elif task_type == "humanseg":
        human_seg = hub.Module(name="humanseg_lite")

    last_time = time.time()
    while True:
        start_time = time.time()
        print("Frame buffer of streaming 1", len(NDNstreaming1.decoded_frame))
        # print("Frame buffer of streaming 2", len(NDNstreaming2.decoded_frame))
        if len(NDNstreaming1.decoded_frame) > 0:
            producer_frame = NDNstreaming1.decoded_frame[-1][1]
            if task_type == "raw":
                display_image = producer_frame
            elif task_type == "yolo":
                # producer_frame = cv2.rotate(, cv2.ROTATE_90_CLOCKWISE)
                # check for common shapes
                s = np.stack([letterbox(producer_frame, new_shape=imgsz)[0].shape], 0)  # inference shapes
                yolo_rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
                if not yolo_rect:
                    print(
                        'WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')
                # Letterbox
                img = [letterbox(producer_frame, new_shape=imgsz, auto=yolo_rect)[0]]
                img = np.stack(img, 0)

                # Convert
                img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                pred = yoloModel(img, augment=args.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=args.classes,
                                           agnostic=args.agnostic_nms)

                # Process detections
                results = []
                for i, det in enumerate(pred):  # detections per image
                    p, s, display_image = "yolo", '', producer_frame.copy()
                    s += '%gx%g ' % img.shape[2:]  # print string
                    # gn = torch.tensor(display_image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if det is not None and len(det):
                        # Rescale boxes from img_size to display_image size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], display_image.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, yoloNames[int(c)])  # add to string

                        # Write results
                        for *xyxy, conf, cls in det:
                            label = '%s %.2f' % (yoloNames[int(cls)], conf)
                            plot_one_box(xyxy, display_image, label=label, color=yoloColors[int(cls)], line_thickness=3)
                            results.append(
                                [yoloNames[int(cls)], str(int(xyxy[0])), str(int(xyxy[1])), str(int(xyxy[2])),
                                 str(int(xyxy[3]))])

                    print("Results: ", results)
                    print('%s Decoding Done. FPS (%.1f)' % (len(results), 1 / (time.time() - start_time)))

                    # if view_img:
                    #     cv2.imshow("Yolo", display_image)
                    #     if cv2.waitKey(1) == ord('q'):  # q to quit
                    #         raise StopIteration
            elif task_type == "face":
                h, w = producer_frame.shape[:-1]
                scale = 640.0 / w
                face_img = cv2.resize(producer_frame, (0, 0), fx=scale, fy=scale)

                # Detect faces
                boxes, _ = mtcnnModel.detect(face_img)

                if boxes is not None:
                    gain = min(face_img.shape[0] / producer_frame.shape[0],
                               face_img.shape[1] / producer_frame.shape[1])  # gain  = old / new
                    pad = (face_img.shape[1] - producer_frame.shape[1] * gain) / 2, (
                            face_img.shape[0] - producer_frame.shape[0] * gain) / 2  # wh padding
                    boxes[:, [0, 2]] -= pad[0]  # x padding
                    boxes[:, [1, 3]] -= pad[1]  # y padding
                    boxes[:, :4] /= gain
                    for box in boxes:
                        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                        cv2.rectangle(producer_frame, c1, c2, [0, 0, 255], thickness=6, lineType=cv2.LINE_AA)
                    display_image = producer_frame.copy()
                else:
                    display_image = producer_frame
            elif task_type == "openpose":
                try:
                    datum = op.Datum()
                    datum.cvInputData = producer_frame
                    opWrapper.emplaceAndPop([datum])
                    display_image = datum.cvOutputData
                except:
                    display_image = producer_frame
            elif task_type == "modelrotate":
                # ======================Add Rotation effect ==========================================
                rotate = trimesh.transformations.rotation_matrix(
                    angle=np.radians(1.0),
                    direction=[0, 1, 1],
                    point=[0, 0, 0])
                # rotate = trimesh.transformations.translation_matrix([0.001,0.001,0])
                pyrenderScene.set_pose(mesh_node, np.dot(pyrenderScene.get_pose(mesh_node), rotate))
                color, depth = offscreenRender.render(pyrenderScene, flags=RenderFlags.RGBA)
                color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA)
                display_image = overlay_transparent_image(producer_frame, color, 0, 0)
                # imgModel = cv2.cvtColor(color, cv2.COLOR_RGBA2RGB)
                # cv2.imshow('imgModel', imgModel)
            elif task_type == "armarker":
                kp2, des2 = orbModel.detectAndCompute(producer_frame, None)
                try:
                    # create BFMatcher object
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                    matches = bf.knnMatch(marker_des1, des2, k=2)
                    good = []
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            good.append(m)
                    print("ARmarker Points Mapping Length: ", len(good))
                    if len(good) > 30:
                        srcPts = np.float32([marker_kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        homoMatrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
                        print(homoMatrix)
                        print("Mask:", len(mask))

                        distance = np.linalg.norm(homoMatrix - last_matrix)
                        if distance < 10:
                            homoMatrix = last_matrix
                        else:
                            last_matrix = homoMatrix
                        print("ARmarker homoMatrix Distance: ", distance)

                        pts = np.float32([[0, 0], [0, markerH], [markerW, markerH], [markerW, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, homoMatrix)

                        background = cv2.polylines(producer_frame, [np.int32(dst)], True, (0, 0, 255), 3)

                        try_pose = np.eye(4)
                        try_pose[0][0] = homoMatrix[0][0]
                        try_pose[0][1] = homoMatrix[0][1]
                        try_pose[1][0] = homoMatrix[1][0]
                        try_pose[1][1] = homoMatrix[1][1]

                        try_pose[3][0] = -homoMatrix[2][0]
                        try_pose[3][1] = -homoMatrix[2][1]
                        # print("Projection", try_pose)

                        pyrenderScene.set_pose(mesh_node, np.dot(np.eye(4), try_pose))
                        color, depth = offscreenRender.render(pyrenderScene, flags=RenderFlags.RGBA)
                        color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA)
                        # cv2.imshow('background', background)
                        # cv2.imshow('color', color)
                        display_image = overlay_transparent_image(background, color, 0, 0)
                    else:
                        display_image = producer_frame
                except Exception as e:
                    print("ARMarker error： ", e)
                    display_image = producer_frame
                # imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)
            elif task_type == "arvideo":
                arVideoFrameCounter += 1
                imgWebcam = producer_frame
                imgAug = imgWebcam.copy()
                kp2, des2 = orbModel.detectAndCompute(imgWebcam, None)

                if markerDetection == False:
                    arVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    arVideoFrameCounter = 0
                else:
                    if arVideoFrameCounter == arVideo.get(cv2.CAP_PROP_FRAME_COUNT):
                        arVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        arVideoFrameCounter = 0
                    success, imgArVideo = arVideo.read()
                    imgArVideo = cv2.resize(imgArVideo, (markerW, markerH))
                try:
                    # create BFMatcher object
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                    matches = bf.knnMatch(marker_des1, des2, k=2)
                    good = []
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            good.append(m)
                    if len(good) > 30:
                        markerDetection = True
                        try:
                            srcPts = np.float32([marker_kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                            dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                            matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)

                            # # Refine the points with mask
                            # srcPts = np.float32([kp1[good[m].queryIdx].pt for m in range(len(good)) if mask[m][0]]).reshape(-1, 1, 2)
                            # dstPts = np.float32([kp2[good[m].trainIdx].pt for m in range(len(good)) if mask[m][0]]).reshape(-1, 1, 2)
                            # matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
                            #
                            # print(matrix)
                            # print("Mask:", len(mask))

                            pts = np.float32([[0, 0], [0, markerH], [markerW, markerH], [markerW, 0]]).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, matrix)
                        except:
                            display_image = producer_frame

                        distance = np.linalg.norm(dst - last_dst)
                        if distance < 20:
                            dst = last_dst
                        else:
                            last_dst = dst
                        print("Distance: ", distance)
                        # img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (0, 0, 255), 3)
                        imgWarp = cv2.warpPerspective(imgArVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))
                        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
                        cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
                        maskInv = cv2.bitwise_not(maskNew)
                        imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
                        display_image = cv2.bitwise_or(imgWarp, imgAug)
                except:
                    display_image = producer_frame
            elif task_type == "humanseg":
                cv2.imshow("Original", producer_frame)
                display_image = human_seg_light(human_seg, producer_frame)

            encoder.stdin.write(
                display_image
                    .astype(np.uint8)
                    .tobytes()
            )
            encoder.stdin.flush()
            print("handle interval: ", time.time() - last_time, " curr_time: ", time.time())
            last_time = time.time()
            cv2.imshow("Output", display_image.astype("uint8"))
            # Need to convert to uint8 for processing and display
            # cv2.imshow("Shareview", left_frame.astype("uint8"))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                # sys.exit(-1)
        # Use a constant interval for fetching the frames
        sleeptime = max(0.0, interval - (time.time() - start_time))
        # print("handle sleeptime: ", sleeptime, " ", time.time() - start_time, " curr_time: ", time.time())
        time.sleep(sleeptime)


def get_frames():
    global current_I_frame, frame_buffer_I, frame_buffer_P, I_frame_index
    # Split the sps, I, P frames
    if device_type == "gpu":
        frame_types = {"SPS": b'\x00\x00\x00\x01g', "PPS": b'\x00\x00\x00\x01h', "SEI": b'\x00\x00\x00\x01\x06',
                       "I": b'\x00\x00\x00\x01e', "P": b'\x00\x00\x00\x01a'}
    else:
        frame_types = {"SPS": b'\x00\x00\x00\x01g', "I": b'\x00\x00\x00\x01h', "P": b'\x00\x00\x00\x01A', "PPS": b'',
                       "SEI": b''}

    total_frames = 0
    start_time = time.time()
    h264_split_key = b"\x00\x00\x00\x01"
    last_frame = b''
    sps_info = b''
    pps_info = b''
    sei_info = b''
    frame_num = 1
    last_time = time.time()
    curr_time = time.time()
    last_time_frame = 0
    frame_type = "I"
    while True:
        content = bytes(encoder.stdout.read(9500))

        # If there are part of the I frames in the frame, put the I flag
        frames = [h264_split_key + x for x in content.split(h264_split_key)]
        frames[0] = frames[0].strip(h264_split_key)
        Iflag = True
        for f in frames:
            if f != b'' and Iflag:
                print('append Parse frame' + str(f[0:10]) + "frame length: " + str(len(f)))
                header = f[0:5]
                if header == frame_types['I']:
                    print("append I frame Interval: ---------------", time.time() - curr_time)
                    curr_time = time.time()
                    frame_type = "I"
                    current_I_frame = frame_num
                    Iflag = False
                elif header == frame_types['P']:
                    frame_type = "P"
                    print("append Append P frame of length {}".format(len(f)), " frame number: ", frame_num)

        frame_buffer.append(frame_num)
        content = encrypt_message(content)
        frame_buffer_dict[frame_num] = (content, frame_type)
        frame_num += 1
        print("handle Append Unified frame of length {}".format(len(content)), " frame number: ", frame_num, "time: ",
              time.time())
        if time.time() - last_time > 1:
            print("handle Append Unified frame FPS:", frame_num - last_time_frame)
            last_time_frame = frame_num
            last_time = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NDN AR demo")
    parser.add_argument('--task', type=str, default='yolo', help='AR Task Type')  # Task
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--weights', nargs='+', type=str, default='models/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    args = parser.parse_args()

    eth = threading.Thread(target=video_encoder)
    eth.start()
    fth = threading.Thread(target=get_frames)
    fth.start()
    app.run_forever()

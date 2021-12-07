import numpy as np
import NDNstreaming1
import threading, time, cv2, ffmpeg, sys, os, argparse, logging
from collections import deque
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA
from ndn.app import NDNApp
from ndn.encoding import Name, InterestParam, BinaryStr, FormalName, Component
from ndn.security import Sha256WithRsaSigner
from security.DefaultKeys import DefaultKeys
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import torch
import torch.backends.cudnn as cudnn
from typing import Optional
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

ffmpeg_type = "gpu"  # "gpu" or "cpu"
height = 1080
width = 1920
crf = 30
interval = 1. / crf
logging.basicConfig(level=logging.DEBUG)

try:
    import thread
except ImportError:
    import _thread as thread

dth = threading.Thread(target=NDNstreaming1.run, args=('NEAR/group1/producer1', width, height, ffmpeg_type))
dth.start()

if ffmpeg_type == "gpu":
    encoder = (
        ffmpeg
            .input("pipe:", format='rawvideo', s='{}x{}'.format(width, height), pix_fmt='bgr24')
            # .output("pipe:", format='h264', vcodec='h264_nvenc', crf=crf, g=crf, keyint_min=crf, bf=0, bitrate='320k', profile="baseline", pix_fmt='yuv420p')  # 720p
            .output("pipe:", format='h264', vcodec='h264_nvenc', crf=crf, g=crf, keyint_min=crf, bf=0, bitrate='2000k',
                    profile="baseline", pix_fmt='yuv420p')  # 1080p
            .video
            .run_async(pipe_stdin=True, pipe_stdout=True)
    )
else:
    encoder = (
        ffmpeg
            .input("pipe:", format='rawvideo', s='{}x{}'.format(width, height), pix_fmt='bgr24')
            .output("pipe:", format='h264', vcodec='libx264', crf=crf, g=crf, keyint_min=crf, bf=0, pix_fmt='yuv420p')
            .video
            .run_async(pipe_stdin=True, pipe_stdout=True)
    )

producer_frame = np.zeros([height, width, 3])
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

# Sign the data packet with the edge key /NEAR/edge/KEY/000
app_keys = DefaultKeys()
private_key = app_keys.private_key
public_key = app_keys.public_key
signer = Sha256WithRsaSigner('/NEAR/edge/KEY/000', private_key)
signer_cert = Sha256WithRsaSigner('/NEAR/edge/KEY', private_key)
# Encryption setting
cipher_key = bytearray(16)
enable_cipher = True


def encrypt_message(plaintext, key=cipher_key):
    if not enable_cipher:
        return plaintext
    iv = os.urandom(16)
    cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=128)
    return iv + cipher.encrypt(plaintext)


@app.route('/NEAR/edge')
def on_interest(name: FormalName, param: InterestParam, _app_param: Optional[BinaryStr]):
    logging.info(f'>> I: {Name.to_str(name)}, {param}')
    request = Name.to_str(name).split("/")
    print("handle Interest Name", Name.to_str(name))
    if request[4] == "metadata":
        print("handle Meta data")
        # content = json.dumps(list(pred_frame_buffer)).encode()
        # content = str(current_I_frame).encode()
        content = Name.to_str(name + [Component.from_number(current_I_frame, 0)]).encode()
        name = name
        app.put_data(name, content=content, freshness_period=300)
        logging.info("handle to name " + Name.to_str(name))
    elif request[4] == "chunk":
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
    elif request[3] == "KEY":
        content = public_key
        name = name
        # The key is signed by the certificate, which is trusted by the mobile device
        app.put_data(name, content=content, freshness_period=3000, signer=signer_cert)
    elif request[3] == "CK":
        # Use the receiver's public key to encrypt the content key
        rsa_public_key = RSA.importKey(public_key)
        rsa_public_key = PKCS1_OAEP.new(rsa_public_key)
        encrypted_ck = rsa_public_key.encrypt(cipher_key)
        content = encrypted_ck
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

@app.route('/ca')
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
        yoloColors = [[np.random.randint(0, 255) for _ in range(3)] for _ in yoloNames]

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
    if ffmpeg_type == "gpu":
        frame_types = {"SPS": b'\x00\x00\x00\x01g', "PPS": b'\x00\x00\x00\x01h', "SEI": b'\x00\x00\x00\x01\x06',
                       "I": b'\x00\x00\x00\x01e', "P": b'\x00\x00\x00\x01a'}
    else:
        frame_types = {"SPS": b'\x00\x00\x00\x01g', "I": b'\x00\x00\x00\x01h', "P": b'\x00\x00\x00\x01A', "PPS": b'',
                       "SEI": b''}

    total_frames = 0
    start_time = time.time()
    h264_split_key = b"\x00\x00\x00\x01"
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

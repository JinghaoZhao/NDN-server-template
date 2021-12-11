from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA
from ndn.security import Sha256WithRsaSigner
from security.DefaultKeys import DefaultKeys
from Crypto.Cipher import AES
# import NDNstreaming1
import NDNstreaming2 as NDNstreaming1
import ffmpeg
from ndn.app import NDNApp
from ndn.encoding import Name, InterestParam, BinaryStr, FormalName, MetaInfo, Component
from collections import deque
from typing import Optional
from queue import Queue
import numpy as np
from tf_pose.estimator import TfPoseEstimator, TfPoseEstimatorSacc
from tf_pose.networks import get_graph_path, model_wh
import threading, os, sys, cv2, signal, time, logging, argparse

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

try:
    sys.path.append('/usr/local/python')
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

ffmpeg_type = "cpu"  # "gpu" or "cpu"
height = 480  # 1080
width = 640  # 1920
crf = 30
interval = 1. / crf
logging.basicConfig(level=logging.DEBUG)

try:
    import thread
except ImportError:
    import _thread as thread

dth = threading.Thread(target=NDNstreaming1.run, args=('NEAR/group1/producer1', width, height, ffmpeg_type))
# dth = threading.Thread(target=NDNstreaming1.run, args=('testecho',))
dth.start()

if ffmpeg_type == "gpu":
    encoder = (
        ffmpeg
            .input("pipe:", format='rawvideo', s='{}x{}'.format(width, height), pix_fmt='bgr24')
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


def test_resize(in_q, out_q):
    while True:
        display_image = in_q.get()

        # resize image
        # display_image = cv2.resize(display_image, (384*2, 384*2), interpolation = cv2.INTER_AREA)
        display_image = cv2.resize(display_image, (width, height), interpolation=cv2.INTER_AREA)
        out_q.put(display_image)


def test_display(in_q):
    while True:
        display_image = in_q.get()

        # resize image
        # display_image = cv2.resize(display_image, (384*2, 384*2), interpolation = cv2.INTER_AREA)
        # print('Received Image')
        in_q.task_done()
        encoder.stdin.write(
            display_image
                .astype(np.uint8)
                .tobytes()
        )
        encoder.stdin.flush()
        # print("handle interval: ", time.time() - last_time, " curr_time: ", time.time())
        last_time = time.time()

        # Need to convert to uint8 for processing and display
        DISPLAY_FLAG = True
        if DISPLAY_FLAG:
            cv2.imshow("ServerDisplay", display_image.astype("uint8"))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            pass
            # cv2.imwrite("ServerDisplay.jpg", display_image.astype("uint8"))
            # out.write(display_image.astype("uint8"))


def test_draw(in_q, out_q):
    while True:
        (producer_frame, humans) = in_q.get()
        display_image = TfPoseEstimator.draw_humans(producer_frame, humans[0], imgcopy=False)
        # display_image = producer_frame
        out_q.put(display_image)


def worker_write(q2):
    import tensorflow as tf
    frame_width = 384
    frame_height = 384
    out = cv2.VideoWriter('output_{}.avi'.format('FPGA'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                          (frame_width, frame_height))
    while True:

        if not q2.empty():
            image6 = q2.get()
            out.write(image6)


def video_encoder(out_q):
    global left_frame, display_image
    # cv2.namedWindow("Output", cv2.WINDOW_GUI_NORMAL)
    # producer_frame = np.random.randint(0, 256, height * width * 3, dtype='uint8')

    task_type = args.task

    last_time = time.time()
    while True:
        start_time = time.time()
        print("Frame buffer of streaming 1", len(NDNstreaming1.decoded_frame))
        # print("Frame buffer of streaming 2", len(NDNstreaming2.decoded_frame))
        if len(NDNstreaming1.decoded_frame) > 0:
            producer_frame = NDNstreaming1.decoded_frame[-1][1]
            humans = fpga_est.inference(producer_frame, resize_to_default=True, upsample_size=args.resize_out_ratio)
            # humans = []
            if not args.showBG:
                producer_frame = np.zeros(producer_frame.shape)
            # display_image = TfPoseEstimator.draw_humans(producer_frame, humans[0], imgcopy=False)
            # out_q.put(display_image)
            out_q.put((producer_frame, humans))

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


def signal_handler(signal, frame):
    os.killpg(os.getpgid(encoder.pid), 9)
    os.killpg(os.getpgid(NDNstreaming1.pid), 9)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NDN AR demo")
    parser.add_argument('--task', type=str, default='raw', help='AR Task Type')  # Task
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--model', type=str, default='mobilenet_thin',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='384x384',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--display', action='store_true',
                        help='whether output result to monitor')
    parser.add_argument('--device', type=str, default='FPGA',
                        help='specify the inference device')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if args.device == 'CPU':
        if w == 0 or h == 0:
            fpga_est = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
        else:
            fpga_est = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    elif args.device == 'FPGA':
        if w == 0 or h == 0:
            fpga_est = TfPoseEstimatorSacc(get_graph_path(args.model), target_size=(432, 368))
        else:
            fpga_est = TfPoseEstimatorSacc(get_graph_path(args.model), target_size=(w, h))

    out_video = Queue()
    out_draw = Queue()
    out_resize = Queue()

    eth = threading.Thread(target=video_encoder, args=(out_video,))
    dth = threading.Thread(target=test_draw, args=(out_video, out_draw))
    rth = threading.Thread(target=test_resize, args=(out_draw, out_resize))
    tth = threading.Thread(target=test_display, args=(out_draw,))
    # wth = threading.Thread(target=worker_write, args=(out_draw,))
    eth.start()
    dth.start()
    # rth.start()
    tth.start()
    # wth.start()

    fth = threading.Thread(target=get_frames)
    fth.start()
    app.run_forever()
    out_video.join()

    signal.signal(signal.SIGINT, signal_handler)
    encoder.wait()
    print('finish')

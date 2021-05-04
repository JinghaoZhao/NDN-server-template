import numpy as np
import NDNstreaming1
import threading
import cv2
import ffmpeg
from ndn.app import NDNApp
from ndn.encoding import Name, InterestParam, BinaryStr, FormalName, MetaInfo, Component
import logging
import time
from collections import deque
import argparse
from typing import Optional
import sys


try:
    sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

device_type = "cpu"  # "gpu" or "cpu"
height = 720
width = 1280
logging.basicConfig(level=logging.DEBUG)

try:
    import thread
except ImportError:
    import _thread as thread

dth = threading.Thread(target=NDNstreaming1.run, args=('testecho',))
dth.start()

crf = 30
interval = 1. / crf
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


@app.route('/edge')
def on_interest(name: FormalName, param: InterestParam, _app_param: Optional[BinaryStr]):
    logging.info(f'>> I: {Name.to_str(name)}, {param}')
    request = Name.to_str(name).split("/")
    print("handle Interest Name", Name.to_str(name))
    if request[-2] == "metadata":
        print("handle Meta data")
        # content = json.dumps(list(pred_frame_buffer)).encode()
        # content = str(current_I_frame).encode()
        content = Name.to_str(name + [Component.from_number(current_I_frame, 0)]).encode()
        name = name
        app.put_data(name, content=content, freshness_period=300)
        logging.info("handle to name " + Name.to_str(name))
    elif request[-3] == "frame":
        interest_frame_num = int(request[-1])
        if interest_frame_num in frame_buffer_dict:
            content = frame_buffer_dict[interest_frame_num]
            app.put_data(name + [b'\x08\x02\x00\x00'], content=content, freshness_period=2000, final_block_id=Component.from_segment(0))
            print(f'handle interest: publish pending interest' + Name.to_str(name) + "------------/" + str(interest_frame_num) + "length: ", len(content))
        else:
            interest_buffer.append([interest_frame_num, name])
    else:
        print("handle Request missing ", Name.to_str(name))

    while len(interest_buffer) > 0 and len(frame_buffer) > 0 and frame_buffer[-1] >= interest_buffer[0][0]:
        pendingInterest = interest_buffer.popleft()
        pendingFN = pendingInterest[0]
        pendingName = pendingInterest[1]
        if pendingFN in frame_buffer_dict:
            content = frame_buffer_dict[pendingFN]
            app.put_data(pendingName + [b'\x08\x02\x00\x00'], content=content, freshness_period=2000, final_block_id=Component.from_segment(0))
            print(f'handle interest: publish pending interest' + Name.to_str(pendingName) + "------------/" + str(pendingFN) + "length: ", len(content))



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


def video_encoder():
    global left_frame, display_image
    # cv2.namedWindow("Output", cv2.WINDOW_GUI_NORMAL)
    # producer_frame = np.random.randint(0, 256, height * width * 3, dtype='uint8')

    task_type = args.task

    if task_type == "raw":
        # TODO: Load ML Models
        pass

    last_time = time.time()
    while True:
        start_time = time.time()
        print("Frame buffer of streaming 1", len(NDNstreaming1.decoded_frame))
        # print("Frame buffer of streaming 2", len(NDNstreaming2.decoded_frame))
        if len(NDNstreaming1.decoded_frame) > 0:
            producer_frame = NDNstreaming1.decoded_frame[-1][1]


            if task_type == "raw":
                # TODO: Porcess the frame with ML model, then put the processed frame as display_image
                display_image = cv2.cvtColor(cv2.Canny(producer_frame, 100, 200), cv2.COLOR_GRAY2BGR)
                # display_image = producer_frame

            encoder.stdin.write(
                display_image
                    .astype(np.uint8)
                    .tobytes()
            )
            encoder.stdin.flush()
            print("handle interval: ", time.time() - last_time, " curr_time: ", time.time())
            last_time = time.time()

            # Need to convert to uint8 for processing and display
            cv2.imshow("ServerDisplay", display_image.astype("uint8"))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
    curr_time = time.time()
    while True:
        content = bytes(encoder.stdout.read(3000))
        # print(bytes(content))
        # Split the content
        content = last_frame + content
        frames = [h264_split_key + x for x in content.split(h264_split_key)]
        frames[0] = frames[0].strip(h264_split_key)
        tmp_frame_num = 0
        for f in frames[0:-1]:
            if f != b'':
                # print('append Parse frame' + str(f[0:10]) + "frame length: " + str(len(f)))
                header = f[0:5]
                if header == frame_types['SPS']:
                    sps_info = f
                    print("handle Append SPS frame of length {}".format(len(f)), f)
                elif header == frame_types['PPS']:
                    pps_info = f
                    print("handle Append PPS frame of length {}".format(len(f)), f)
                elif header == frame_types['SEI']:
                    sei_info = f
                    print("handle Append PPS frame of length {}".format(len(f)), f)
                elif header == frame_types['I']:
                    print("handle I frame Interval: ---------------", time.time() - curr_time)
                    curr_time = time.time()
                    current_I_frame = frame_num
                    I_frame_index = current_I_frame % 30
                    frame_buffer.append(frame_num)
                    frame_buffer_dict[frame_num] = (sps_info + pps_info + sei_info + f)
                    print("handle Append I frame of length {}".format(len(sps_info + pps_info + sei_info + f)),
                          " frame number: ", frame_num, time.time())
                    frame_num += 1
                elif header == frame_types['P']:
                    frame_buffer.append(frame_num)
                    frame_buffer_dict[frame_num] = (sei_info + f)
                    print("handle Append P frame of length {}".format(len(sei_info + f)), " frame number: ", frame_num)
                    frame_num += 1
                # else:
                #     logging.warning('Wrong frame' + str(f[0:10]) + "frame length: " + str(len(f)))
                tmp_frame_num += 1
        last_frame = frames[-1]
        total_frames += tmp_frame_num
        current_time = time.time()
        logging.info("Current byte rate is {:.2f}".format(1. * len(content) / (current_time - start_time)))
        logging.info("Current encoding frame rate is {:.2f}".format(1. * tmp_frame_num / (current_time - start_time)))
        start_time = current_time
        logging.info("Encoded {} frames".format(total_frames))
        # print("Frame types: ", frame_types)
        # time.sleep(0.02)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NDN AR demo")
    parser.add_argument('--task', type=str, default='raw', help='AR Task Type')  # Task
    args = parser.parse_args()

    eth = threading.Thread(target=video_encoder)
    eth.start()
    fth = threading.Thread(target=get_frames)
    fth.start()
    app.run_forever()

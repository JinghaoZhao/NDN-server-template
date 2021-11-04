import numpy as np
import NDNstreaming1
import NDNstreaming2
import threading
import cv2
import ffmpeg
from typing import Optional
from ndn.app import NDNApp
from ndn.encoding import Name, InterestParam, BinaryStr, FormalName, MetaInfo, Component
import logging
import time
from collections import deque
import shareview.ImageStitching as imgst
try:
    import thread
except ImportError:
    import _thread as thread

ffmpeg_type = "gpu"
height = 1080
width = 1920
crf = 30
interval = 1. / crf
logging.basicConfig(level=logging.DEBUG)

dth = threading.Thread(target=NDNstreaming1.run, args=('NEAR/group1/producer1',))
dth2 = threading.Thread(target=NDNstreaming2.run, args=('NEAR/group1/producer2',))
dth.start()
dth2.start()


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

left_frame = np.zeros([height, width, 3])
right_frame = np.zeros([height, width, 3])
# display_image = np.hstack((left_frame, right_frame))
display_image = left_frame
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
            app.put_data(name + [b'\x08\x02\x00\x00'], content=content, freshness_period=5000, final_block_id=Component.from_segment(0))
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
            app.put_data(pendingName + [b'\x08\x02\x00\x00'], content=content, freshness_period=5000, final_block_id=Component.from_segment(0))
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


def video_encoder():
    global left_frame, display_image
    stitching = imgst.ImageStitching()
    start_time = time.time()
    cv2.namedWindow("Shareview", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow("RawVideo", cv2.WINDOW_GUI_NORMAL)
    left_frame = np.random.randint(0, 256, height * width * 3, dtype='uint8')
    right_frame = np.random.randint(0, 256, height * width * 3, dtype='uint8')
    while True:
        print("Frame buffer of streaming 1", len(NDNstreaming1.decoded_frame))
        print("Frame buffer of streaming 2", len(NDNstreaming2.decoded_frame))
        if len(NDNstreaming1.decoded_frame) > 0:
            left_frame = NDNstreaming1.decoded_frame[-1][1]
            st_left = left_frame.copy()
            h, w = st_left.shape[:-1]
            scale = (width / 2) / w
            # Need to convert to uint8 for processing and display
            st_left = np.uint8(cv2.resize(st_left, (0, 0), fx=scale, fy=scale))
            stitching.img1 = st_left.copy()
            cv2.imshow("Producer1", left_frame)
        if len(NDNstreaming2.decoded_frame) > 0:
            right_frame = NDNstreaming2.decoded_frame[-1][1]
            st_right = right_frame.copy()
            h, w = st_right.shape[:-1]
            scale = (width / 2) / w
            st_right = np.uint8(cv2.resize(st_right, (0, 0), fx=scale, fy=scale))
            stitching.img2 = st_right.copy()
            cv2.imshow("Producer2", right_frame)
        try:
            display_image = np.hstack((left_frame, right_frame))
        except:
            display_image = np.zeros(height * width * 3)
        sharedview = stitching.update(1)
        if sharedview is None:
            print("Waiting for shareview...")
            sharedview = display_image
        else:
            sharedview = resizeImg(sharedview, width, height)
            # left_frame = np.random.randint(0, 256, height*width*3, dtype='uint8')
            # left_frame = np.zeros(height * width * 3)
            # display_image = left_frame
            encoder.stdin.write(
                sharedview
                    .astype(np.uint8)
                    .tobytes()
            )
            encoder.stdin.flush()
        cv2.imshow("RawVideo", display_image)
        # Need to convert to uint8 for processing and display
        cv2.imshow("Shareview", sharedview.astype("uint8"))
        cv2.waitKey(1)
        # Use a constant interval for fetching the frames
        time.sleep(interval - (time.time() - start_time) % interval)

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
    eth = threading.Thread(target=video_encoder)
    eth.start()
    fth = threading.Thread(target=get_frames)
    fth.start()
    app.run_forever()

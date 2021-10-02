import logging
import time

import ndn.utils
# from ndn.app import NDNApp
from MyNDNApp import MyNDNApp
from ndn.types import InterestNack, InterestTimeout, InterestCanceled, ValidationFailure
from ndn.encoding import Name, Component, InterestParam, BinaryStr, FormalName, MetaInfo

from frame_fetcher import frame_fetcher
import asyncio
from asyncio import Future
import numpy as np
from numpy import random
import ffmpeg
import cv2
import io
import os
import shutil
from security.checkers import sha256_rsa_checker
from Crypto.Cipher import AES

# None block read
import fcntl

# Use the receiver and decoder in separate threads
import threading
from collections import deque

try:
    import thread
except ImportError:
    import _thread as thread

logging.basicConfig()
app = MyNDNApp()
app.data_validator = sha256_rsa_checker

view_img = False

header_dict = {"I": b'\x00\x00\x00\x01g', "P": b'\x00\x00\x00\x01A'}
height = 720
width = 1280
delay = 0
ct = 0
dt = []
start_feed = False
in_frame_num = 0
out_frame_num = 0
lost_frames = 0

"""
frame_queue is to maintain the interest window:
1. When the queue is shorter than the window_len, the NDN consumer sends interest to the producer and append 
   [global_frame_num, None] to the right of the queue
2. When the producer response with the content, the consumer update thew content in the queue with [global_frame_num, content]
   - If the (left) head of the queue is not empty, the queue pops the head of the queue and sends the content to the 
   on_frame_message().
   - If the received frame has a large gap with the head, we assume that this frame can never be retrieved, so pop it out 
   of the queue. 
   - *If the missed frame is an I frame, the following frames should all be deprecated until receiving the next I frame.
"""
window_len = 50
frame_queue = deque()
in_frame_dict = {}  # Match the in_frame_num/out_frame_num: global_frame_num
pred_dict = {}  # global_frame_num: result
pred_frame_buffer = deque(maxlen=50)  # store the most recent results
yoloInterestQueue = deque()
decoded_frame = deque(maxlen=5)
enable_cipher = True
cipher_key = bytes(16)
decryption_timer = deque(maxlen=300)


def decrypt_message(ciphertext, key=cipher_key):
    if not enable_cipher:
        return ciphertext
    # start_time = time.time()
    iv = ciphertext[:16]
    cipher = AES.new(key, AES.MODE_CFB, iv, segment_size=128)
    plaintext = cipher.decrypt(ciphertext[16:])
    # decryption_timer.append(time.time() - start_time)
    # print("Average decryption time {}".format(np.mean(decryption_timer)))
    # print("STD decryption time {}".format(np.std(decryption_timer)))
    return plaintext


async def get_last_frame(device_name):
    """
    Get the last frame number from interest /device/1080p/metadata/timestamp
    """
    message_counter = 0
    while message_counter < 100:
        try:
            timestamp = ndn.utils.timestamp()
            name = Name.from_str('/{}/1080p/metadata/'.format(device_name)) + [Component.from_timestamp(timestamp)]
            print(f'Sending Interest {Name.to_str(name)}, {InterestParam(must_be_fresh=False, lifetime=600)}')
            data_name, meta_info, content = await app.express_interest(
                name, must_be_fresh=False, can_be_prefix=True, lifetime=2000)
            print(f'Received Data Name: {Name.to_str(data_name)}')
            ct = bytes(content)
            last_frame_name = Name.from_str(str(ct)[2:-1])
            print("Last Frame number ", Name.to_str(last_frame_name))
            last_frame_num = Component.to_number(last_frame_name[-1])
            return last_frame_num
        except InterestNack as e:
            print(f'Nacked with reason={e.reason}')
        except InterestTimeout:
            print(f'Timeout')
        except InterestCanceled:
            print(f'Canceled')
        except ValidationFailure:
            print(f'Data failed to validate')
        time.sleep(0.03)
        message_counter += 1


async def main(device_name, process):
    global lost_frames, start_feed, frame_queue
    received_frame_num = 0

    def frame_callback(future):
        nonlocal received_frame_num
        global lost_frames
        if future.result() is not None:
            global_frame_num, frame_content = future.result()

            frame_content = decrypt_message(frame_content)
            print("Frames {} fetched with {} bytes".format(global_frame_num, len(frame_content)))
            received_frame_num += 1
            if len(frame_queue) > 0 and frame_queue[0][0] <= global_frame_num:
                # In case the queue has been refreshed
                # Update the frame queue
                print("Stream 1: Received frame num {}, frame head in queue {}".format(global_frame_num,
                                                                                       frame_queue[0][0]))
                print("Stream 1: Frames in the queue: ", [x[0] for x in frame_queue])
                frame_queue[global_frame_num - frame_queue[0][0]][1] = frame_content

                # on_frame_message(frame_content, global_frame_num)
                if global_frame_num - frame_queue[0][0] > 10:
                    # pop out the empty frames
                    while frame_queue[0][1] is None:
                        frame_queue.popleft()

                while (len(frame_queue) > 0) and (frame_queue[0][1] is not None):
                    head_frame = frame_queue.popleft()
                    on_frame_message(head_frame[1], head_frame[0], process)

        else:
            lost_frames += 1

    reconnect_time = 0
    while reconnect_time < 1000:
        reconnect_time += 1
        lost_frames = 0
        last_frame_num = await get_last_frame(device_name)
        if last_frame_num is None:
            print("Get last frame None ")
        loop = asyncio.get_running_loop()
        # asyncio.set_event_loop(loop)
        futures = []
        start_time = time.time()
        current_frame_num = last_frame_num
        iframe_index = last_frame_num % 30
        while (lost_frames <= 20) and (current_frame_num < last_frame_num + 10000):
            # Just in case, refreshes the last frame number in every 10000 frames
            if len(frame_queue) < window_len:
                # h264_results_name = "/{}/1080p/frame/".format(device_name) + str(current_frame_num)
                if current_frame_num % 30 == iframe_index:
                    h264_results_name = "/{}/1080p/frame/I/{}".format(device_name, current_frame_num)
                else:
                    h264_results_name = "/{}/1080p/frame/P/{}".format(device_name, current_frame_num)
                print("Requesting frame", h264_results_name)
                future = loop.create_future()
                future.add_done_callback(frame_callback)
                loop.create_task(frame_fetcher(app, h264_results_name, future, global_frame_num=current_frame_num))
                futures.append(future)
                frame_queue.append([current_frame_num, None])
                current_frame_num += 1
            await asyncio.sleep(0.015 - ((time.time() - start_time) % 0.015))
        print("Reconnecting")
        frame_queue.clear()
        start_feed = False
        await asyncio.wait(futures, timeout=60)
        print("Receive frames: ", received_frame_num)
    app.shutdown()


def on_frame_message(message, global_frame_num, process):
    """
    When the message arrives, write the input to the stdin of the ffmpeg processor
    Start feeding from the first I frame
    """
    global dt, ct, start_feed, in_frame_num, lost_frames
    print("Receive message, length {} bytes".format(len(message)))

    if message[:5] == header_dict["I"]:
        start_feed = True

    if start_feed:
        process.stdin.write(message)
        process.stdin.flush()
        in_frame_dict[in_frame_num] = global_frame_num
        in_frame_num += 1
        print("Feed frames for decoding", global_frame_num, "; in frame:", in_frame_num)


def display_thread(process, device_name, width, height):
    global in_frame_num, out_frame_num, view_img
    i = 0
    print("New Thread for display")

    """
    Load the yolo module
    """

    jump_frame_threshold = 1
    while True:
        i += 1
        if in_frame_num > out_frame_num:
            t1 = time.time()
            print("Stream1: Decoding frame ", out_frame_num, "; Current in_frame is ", in_frame_num,
                  "; in_frame_dic Size:",
                  len(in_frame_dict))
            in_bytes = process.stdout.read(width * height * 3 * jump_frame_threshold)
            # print('%s Decoding. time--------------(%.5f)' % ("Read", (time.time() - t1)))
            if not in_bytes:
                print("No bytes available")
                break
            in_frame = (
                np
                    .frombuffer(in_bytes, np.uint8)
                    .reshape([-1, height, width, 3])
            )
            global_frame_num = in_frame_dict[out_frame_num]
            decoded_frame.append([global_frame_num, in_frame[-1]])
            # print("Decoding Global frame ", global_frame_num)
            out_frame_num += jump_frame_threshold
            if view_img:
                cv2.imshow("Stream 1", in_frame[-1])
                cv2.waitKey(1)
        else:
            # print("Jump Decoding frame ", out_frame_num, "; Current in_frame is ", in_frame_num)
            time.sleep(.01)


def run(device_name, width=1920, height=1080, video_name="1080ptest.mp4", ffmpeg_type="cpu", display=False):
    global view_img
    view_img = display
    # Decoding process
    if ffmpeg_type == "cpu":
        process = (
            ffmpeg
            .input('pipe:', format="h264")
            .video
            # .output('captures/output.avi')
            # .output('captures/out1.bgr', format='rawvideo', pix_fmt='bgr24')
            .output("pipe:", format='rawvideo', pix_fmt='bgr24')
            # .global_args('-fflags', 'nobuffer')
            # .global_args('-flags', 'low_delay')
            # .global_args('-avioflags', 'direct')
            # .global_args('-threads', '1')
            # .global_args('-bufsize', '')
            .run_async(pipe_stdin=True, pipe_stdout=True)
        )
    else:
        process = (
            ffmpeg
                .input('pipe:', format="h264", vcodec='h264_cuvid')
                .video
                # .output('captures/output.avi')
                # .output('captures/out1.bgr', format='rawvideo', pix_fmt='bgr24')
                .output("pipe:", format='rawvideo', pix_fmt='bgr24')
                # .global_args('-fflags', 'nobuffer')
                # .global_args('-flags', 'low_delay')
                # .global_args('-avioflags', 'direct')
                # .global_args('-threads', '1')
                # .global_args('-bufsize', '')
                .run_async(pipe_stdin=True, pipe_stdout=True)
        )
    dth = threading.Thread(target=display_thread, args=(process, device_name, width, height))
    dth.start()
    app.run_forever(after_start=main(device_name, process))


if __name__ == '__main__':
    run("testecho", display=True)

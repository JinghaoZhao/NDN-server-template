import time

from ndn.encoding import NonStrictName, Name, Component
from ndn.app import NDNApp
from ndn.types import InterestTimeout
import asyncio
from asyncio import Future


async def frame_fetcher(app: NDNApp, name: NonStrictName, future: Future, global_frame_num=0,
                        validator=None, must_be_fresh=True):
    """
    An async-generator to fetch a segmented object. The first interest fetches the final block id, and the remaining
    interests are issued all together.

    :param app: NDN Application
    :param name: Name prefix of Data
    :param future: the task for getting this frame
    :param global_frame_num: the frame number it's receiving
    :param timeout: Timeout value, in milliseconds
    :param retry_times: Times for retry
    :param validator: Validator
    :param must_be_fresh: MustBeFresh field of Interest
    :return: Data segments in order.
    """

    frame_content = [b'']
    metadata = None
    failure_flag = False

    async def fetch_segment(first, interest, seg_num, timeout=4000,  retry_times=0):
        nonlocal failure_flag
        nonlocal name, metadata, frame_content
        trial_times = 0
        while trial_times <= retry_times:
            trial_times += 1
            # print("Requesting Seg name", Name.to_str(interest_name))
            try:
                name, metadata, content = await app.express_interest(interest, validator=validator,
                                                                     can_be_prefix=first,
                                                                     must_be_fresh=must_be_fresh, lifetime=timeout)
                frame_content[seg_num] = bytes(content)
                break
            except InterestTimeout:
                if trial_times > retry_times:
                    print("Timeout {}: frame {} seg {}".format(trial_times, global_frame_num, seg_num))
                    failure_flag = True

    name = Name.normalize(name)
    # First Interest
    await fetch_segment(True, name, 0)
    if metadata is None:
        # Failed to get the first frame
        print("Frame {} is missing.".format(global_frame_num))
        future.set_result(None)
        return

    finalBlockId = int(bytes(metadata.final_block_id)[-1])
    if finalBlockId > 0:
        # if these are more than one segments
        frame_content.extend([b""] * finalBlockId)
        tasks = []
        for i in range(1, finalBlockId + 1):
            name[-1] = b'\x08\x02\x00' + bytes([i])
            interest_name = name.copy()
            tasks.append(fetch_segment(False, interest_name, i, timeout=200, retry_times=5))
        await asyncio.wait(tasks)

    if not failure_flag:
        future.set_result([global_frame_num, b''.join(frame_content)])
        # print("framefetcher: ", b''.join(frame_content)[0:40])
    else:
        print("Frame {} is missing.".format(global_frame_num))
        future.set_result(None)


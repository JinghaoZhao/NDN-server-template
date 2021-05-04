"""
A customized NDNAPP that can run on the thread other than the main thread.
"""

from ndn.app import NDNApp
from typing import Optional, Any, Awaitable, Coroutine, Tuple, List
import asyncio as aio


class MyNDNApp(NDNApp):
    def run_forever(self, after_start: Awaitable = None) -> bool:
        """
        A non-async wrapper of :meth:`main_loop`.
        Update: enable the app to run on new threads
        :param after_start: the coroutine to start after connection to NFD is established.

        :examples:
            .. code-block:: python3

                app = NDNApp()

                if __name__ == '__main__':
                    app.run_forever(after_start=main())
        """
        task = self.main_loop(after_start)
        try:
            aio.new_event_loop().run_until_complete(task)
            ret = True
        except KeyboardInterrupt:
            logging.info('Receiving Ctrl+C, shutdown')
            ret = False
        finally:
            self.face.shutdown()
        logging.debug('Face is down now')
        return ret

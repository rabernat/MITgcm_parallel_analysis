import numpy as np
from IPython.parallel import Client
from itertools import imap
import llc_worker

LLC = llc_worker.LLCModel()

c = Client(profile='default')
dview = c.direct_view()


@dview.remote(block=True)
def getpid():
    import os
    return os.getpid()
    

tiles = []
for tile in LLC.get_tile_factory():
    tiles.append(tile)

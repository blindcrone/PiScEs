import numpy as np
from PIL import Image
from sys import argv
from time import time
import pisces
proc = pisces.PiScEs(4, 0x4b, 128)
for x in argv[1:]:
    with Image.open(x) as im:
        cr = im.crop((0, 0, 1024, 1008))
        t0 = time()
        xst = proc(np.asarray(cr))
        t = time() - t0
        print(f"{x}: {xst} [{t * 1000}ms]")

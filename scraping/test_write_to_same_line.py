import sys
import time

for i in range(0,2000):
    sys.stdout.write('\rnumber is %d' %i)
    sys.stdout.flush()
    time.sleep(1)

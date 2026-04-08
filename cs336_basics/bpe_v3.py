import regex as re
from collections import defaultdict
import multiprocessing as mp
import time
import argparse



CHUNK_SIZE = 1024 *  50
N_BYTES = 256
# threads num
NUM_COUNTER_PROCESS = 8
NUM_MERGER_PROCESS = 1

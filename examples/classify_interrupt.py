import argparse
import time

import pigpio as pg
import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


# Initialise output GPIO (2)
p_out = pg.pi()
p_out.set_mode(2, pg.OUTPUT) 
p_out.write(2, 0)

# Initialise input GPIO (26)
p_in = pg.pi()
p_in.set_mode(26, pg.INPUT)

p_stop = pg.pi()
p_stop.set_mode(19, pg.INPUT)

# Parsing arguments from command line to run inference
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-m', '--model', required=True, help='File path of .tflite file.')
parser.add_argument(
    '-i', '--input', required=True, help='Image to be classified.')
parser.add_argument(
    '-l', '--labels', help='File path of labels file.')
parser.add_argument(
    '-k', '--top_k', type=int, default=1,
    help='Max number of classification results')
parser.add_argument(
    '-t', '--threshold', type=float, default=0.0,
    help='Classification score threshold')
parser.add_argument(
    '-c', '--count', type=int, default=5,
    help='Number of times to run inference')
parser.add_argument(
    '-a', '--input_mean', type=float, default=128.0,
    help='Mean value for input normalization')
parser.add_argument(
    '-s', '--input_std', type=float, default=128.0,
    help='STD value for input normalization')
args = parser.parse_args()

labels = read_label_file(args.labels) if args.labels else {}

# LED/Oscilloscope response to indicate the start of making the interpreter
for i in range(2):
    p_out.write(2, 1)
    time.sleep(0.05)
    p_out.write(2, 0)

interpreter = make_interpreter(*args.model.split('@'))
interpreter.allocate_tensors()

size = common.input_size(interpreter)
image = Image.open(args.input).convert('RGB').resize(size, Image.ANTIALIAS)
common.set_input(interpreter, image) # trading for quant and normals

def run_inference():
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    classes = classify.get_classes(interpreter, args.top_k, args.threshold)
    print('%.1fms' % (inference_time * 1000))

# Running the first inference
print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')
run_inference()

# Waiting for GPIO input
calling = p_in.event_callback(26, pg.FALLING_EDGE, run_inference())

print('----Waiting for input----')
while True:
    time.sleep(0.03)
# print('-------RESULTS--------')
# for c in classes:
#     print('%s: %.5f' % (labels.get(c.id, c.id), c.score))




import argparse
import time

import RPi.GPIO as GPIO
import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

# Initialise
BUTTON_PIN = 26
GPIO.setmode(GPIO.BCM)

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

def run_inference(channel):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    classes = classify.get_classes(interpreter, args.top_k, args.threshold)
    print('%.1fms' % (inference_time * 1000))

labels = read_label_file(args.labels) if args.labels else {}

interpreter = make_interpreter(*args.model.split('@'))
interpreter.allocate_tensors()

size = common.input_size(interpreter)
image = Image.open(args.input).convert('RGB').resize(size, Image.ANTIALIAS)
common.set_input(interpreter, image)

print('----INFERENCE TIME----')
print('Note: The first inference on Edge TPU is slow because it includes',
    'loading the model into Edge TPU memory.')
run_inference()
print("---Waiting for input---")

try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    print("Cleaning up...")
    GPIO.cleanup()
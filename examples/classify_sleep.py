import argparse
import time

import pigpio as pg
import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def main():
  
  p1 = pg.pi()
  p1.set_mode(2, pg.OUTPUT) # GPIO 2 as output
  p1.write(2, 0)
  
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

  # First signal for notifying of interpreter making
  p1.write(2, 1)
  time.sleep(0.05)
  p1.write(2, 0)

  interpreter = make_interpreter(*args.model.split('@'))
  interpreter.allocate_tensors()

  # Model must be uint8 quantized
  # if common.input_details(interpreter, 'dtype') != np.uint8:
  #   raise ValueError('Only support uint8 input type.')

  size = common.input_size(interpreter)
  image = Image.open(args.input).convert('RGB').resize(size, Image.ANTIALIAS)
  common.set_input(interpreter, image) # trading for quant and normals

  # Image data must go through two transforms before running inference:
  # 1. normalization: f = (input - mean) / std
  # 2. quantization: q = f / scale + zero_point
  # The following code combines the two steps as such:
  # q = (input - mean) / (std * scale) + zero_point
  # However, if std * scale equals 1, and mean - zero_point equals 0, the input
  # does not need any preprocessing (but in practice, even if the results are
  # very close to 1 and 0, it is probably okay to skip preprocessing for better
  # efficiency; we use 1e-5 below instead of absolute zero).
  # params = common.input_details(interpreter, 'quantization_parameters')
  # scale = params['scales']
  # zero_point = params['zero_points']
  # mean = args.input_mean
  # std = args.input_std
  # if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
  #   # Input data does not require preprocessing.
  #   common.set_input(interpreter, image)
  # else:
  #   # Input data requires preprocessing
  #   normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
  #   np.clip(normalized_input, 0, 255, out=normalized_input)
  #   common.set_input(interpreter, normalized_input.astype(np.uint8))

  # Run inference
  print('----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.')
  for _ in range(args.count):
    p1.write(2, 1)
    time.sleep(0.05)
    p1.write(2, 0)
    time.sleep(0.05)
    
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    classes = classify.get_classes(interpreter, args.top_k, args.threshold)
    print('%.1fms' % (inference_time * 1000))

  print('-------RESULTS--------')
  for c in classes:
    print('%s: %.5f' % (labels.get(c.id, c.id), c.score))


if __name__ == '__main__':
  main()

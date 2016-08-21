# Raymond Cano, Rice University, COMP 540
#
# This script aims to train models based on a validation set and training set.
# The previous Tensorflow tutorial measured it's success on test data, which is
# a terrible thing to do when building a model.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10_input
import cifar10
import cifar10_eval
import numpy as np

VAL_FRAC = .2
USE_GPUS = True

def run():
	global USE_GPUS
	# val_num = np.random.randint(1, 6)
	# cifar10_input.set_constants(train=True, val_num=val_num)
	# if USE_GPUS:			
	# 	import cifar10_multi_gpu_train
	# 	cifar10_multi_gpu_train.main()
	# else:
	# 	import cifar10_train
	# 	cifar10_train.main()
	cifar10_input.set_constants(train=False, val_num=1)
	cifar10_eval.main()

if __name__ == '__main__':
	run()




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import aggregation
import deep_cnn
import input  # pylint: disable=redefined-builtin
import metrics
import numpy as np
from six.moves import xrange
import tensorflow as tf



FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('dataset', 'svhn', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')

tf.flags.DEFINE_string('data_dir','/tmp','Temporary storage')
tf.flags.DEFINE_string('train_dir','/tmp/train_dir','Where model chkpt are saved')
tf.flags.DEFINE_string('teachers_dir','/tmp/train_dir',
                       'Directory where teachers checkpoints are stored.')

tf.flags.DEFINE_integer('teachers_max_steps', 3000,
                        'Number of steps teachers were ran.')
tf.flags.DEFINE_integer('max_steps', 3000, 'Number of steps to run student.')
tf.flags.DEFINE_integer('nb_teachers', 10, 'Teachers in the ensemble.')
tf.flags.DEFINE_integer('stdnt_share', 1000,
                        'Student share (last index) of the test data')
tf.flags.DEFINE_integer('lap_scale', 10,
                        'Scale of the Laplacian noise added for privacy')
tf.flags.DEFINE_boolean('save_labels', False,
                        'Dump numpy arrays of labels and clean teacher votes')
tf.flags.DEFINE_boolean('deeper', False, 'Activate deeper CNN model')



# TODO make generic! For any dataset, any number of classes, etc...?



nb_teachers = 250
nb_classes = 10


def main(argv=None):

  # Load the test dataset from MNIST
  test_data, test_labels = input.ld_mnist(test_only=True) # DATA_DIR?

  # Compute shape of array that will hold probabilities produced by each
  # teacher, for each training point, and each output class
  result_shape = (nb_teachers, len(test_data), nb_classes)

  # Create array that will hold result
  result = np.zeros(result_shape, dtype=np.float32)

  # Get predictions from each teacher
  for teacher_id in xrange(nb_teachers):
  
    # Compute path of checkpoint file for teacher model with ID teacher_id
    ckpt_path = "../RESULTS/MNIST_250/TRAIN_DIR/mnist_250_teachers_"+str(teacher_id)+".ckpt-2999"

    # Get predictions on our training data and store in result array
    preds_for_teacher = deep_cnn.softmax_preds(test_data, ckpt_path)

    # This can take a while when there are a lot of teachers so output status
    print("Computed Teacher " + str(teacher_id) + " softmax predictions")

    # Save in a numpy array
    np.save("PREDOS/predictions_teacher_"+str(teacher_id)+".npy", preds_for_teacher)

  return True
  

if __name__ == "__main__":
  tf.app.run()

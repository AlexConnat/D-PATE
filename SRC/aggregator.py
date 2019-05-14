import tensorflow as tf
import numpy as np
import os
import sys

from agg_utils import parse_params




LAP_SCALE = 20

T = 300
SIGMA1 = 200
SIGMA2 = 40


NB_TEACHERS = 250
NB_SAMPLES = 26032
NB_CLASSES = 10
PREDICTIONS_FOLDER = "/home/ubuntu/D-PATE/RESULTS/SVHN_250/PREDICTIONS"


#######################################################################


def main():

    # Array containing for each teacher, the predicted class for each data sample (integer value for svhn) # string for CIFAR-10?
    predictions = np.zeros((NB_TEACHERS, NB_SAMPLES), dtype=int)


    for teacher_id in range(0,250):

        raw_predictions_teacher_i = np.load(os.path.join(PREDICTIONS_FOLDER, "predictions_teacher_"+str(teacher_id)+".npy"))

        predictions_teacher_i = raw_predictions_teacher_i.argmax(axis=1) # Take the argmax --> [0.001, 0.004, 98.12, 0.000005, 0.003, 0.17, 0.0000114] will collapse to "2"

        # Why not adding up the raw predictions of teachers themselves, and not the final predictions? (argmax)

        predictions[teacher_id] = predictions_teacher_i


    print(predictions.shape)




    #######################
    # Compiling the votes #
    #######################

    votes = np.zeros((NB_SAMPLES, NB_CLASSES), dtype=int)

    for i in range(NB_SAMPLES):

        # Array containing the teachers "votes" (e.g the number of teachers that voted for each class)
        votes_for_sample_i = np.zeros(NB_CLASSES, dtype=int)

        for j in range(NB_TEACHERS):
            votes_for_sample_i[ predictions[j][i] ] += 1

        votes[i] = votes_for_sample_i


    print(votes.shape)
    print(votes)

    #############################################################################


    ############
    # No noise #
    ############


    labels_no_noise = np.zeros(NB_SAMPLES, dtype=int)

    for i in range(NB_SAMPLES):
        labels_no_noise[i] = np.argmax(votes[i])


    print(labels_no_noise.shape)
    print(labels_no_noise)



    ###################
    # Laplacian Noise #
    ###################

    labels_lnmax = np.zeros(NB_SAMPLES, dtype=int)

    for i in range(NB_SAMPLES):
        labels_lnmax[i] = np.argmax( votes[i] + np.random.laplace(0, LAP_SCALE, NB_CLASSES) )


    print(labels_lnmax.shape)
    print(labels_lnmax)

    print(np.sum(labels_no_noise - labels_lnmax))


    ####################
    # Confident GN-MAX #
    ####################

    labels_cgnmax = []
    indices_answered = []
    indices_unanswered = []

    for i in range(NB_SAMPLES):

        if ( np.max(votes[i]) + np.random.normal(0, SIGMA1) >= T ):
            labels_cgnmax.append(  np.argmax( votes[i] + np.random.normal(0, SIGMA2, NB_CLASSES) )  )
            indices_answered.append(i)
        else:
            indices_unanswered.append(i)


    print(len(labels_cgnmax))
    print(len(indices_answered))

    print(labels_cgnmax)
    print(indices_answered)

    print(len(indices_unanswered))



if __name__ == "__main__":

    print( parse_params() )

import numpy as np
import os
import sys

from agg_utils import parse_params, bcolors


#######################################################################


def collect_teachers_predictions(predictions_folder):

    # Gather all filenames of predictions .npy files
    predictions_files = [ f for f in os.listdir(predictions_folder) if 'predictions_teacher_' in f ]

    # How many of these files gives us the number of teachers
    nb_teachers = len(predictions_files)

    if nb_teachers == 0:
        print(bcolors.RED + "Error:" + bcolors.ENDC + " The predictions folder should contain predictions files named 'ptedictions_teacher_i.npy' with 'i' being the teacher ID.")
        exit(-1)

    # We can deduce the number of test samples and number of classes of the ML task from the shape of those predictions
    ex_teacher = np.load( os.path.join(predictions_folder, predictions_files[0]) )
    nb_samples = ex_teacher.shape[0]
    nb_classes = ex_teacher.shape[1]

    # Array containing for each teacher, the predicted class (class ID) for each data sample
    teachers_predictions = np.zeros((nb_teachers, nb_samples), dtype=int)

    # Store each numpy array in a bigger numpy array (indices of predictions files are not neccesarily in order)
    for teacher_id, teacher_preds_file_i in enumerate(predictions_files):
        raw_predictions_teacher_i = np.load( os.path.join(predictions_folder, teacher_preds_file_i) )
        predictions_teacher_i = raw_predictions_teacher_i.argmax(axis=1) # Take the argmax --> [0.001, 0.004, 98.12, 0.000005, 0.003, 0.17, 0.0000114] will collapse to "2"
        teachers_predictions[teacher_id] = predictions_teacher_i
        # TODO: Why not adding up the raw predictions of teachers themselves, and not the final predictions? (argmax)

    return nb_teachers, nb_samples, nb_classes, teachers_predictions # Returns one-hot encoded teachers_predictions



def compile_teachers_votes_from_predictions(nb_teachers, nb_samples, nb_classes, teachers_predictions):

    assert teachers_predictions.shape[0] == nb_teachers
    assert teachers_predictions.shape[1] == nb_samples

    # Array contining for each sample, how many teacher voted for a particular class ID
    teachers_votes = np.zeros((nb_samples, nb_classes), dtype=int)

    for sample_id in range(nb_samples):

        # Array containing the teachers "votes" for one sample (e.g the number of teachers that voted for each class)
        votes_for_sample_i = np.zeros(nb_classes, dtype=int)

        # Increment the right class ID
        for teacher_id in range(nb_teachers):
            votes_for_sample_i[ teachers_predictions[teacher_id][sample_id] ] += 1

        teachers_votes[sample_id] = votes_for_sample_i

    return teachers_votes



#######################################################################


def aggregate_no_noise(teachers_votes):

    nb_samples = teachers_votes.shape[0]
    labels_no_noise = np.zeros(nb_samples, dtype=int)

    for i in range(nb_samples):
        labels_no_noise[i] = np.argmax(teachers_votes[i])

    return labels_no_noise


def aggregate_ln_max(teachers_votes, gamma):

    nb_samples = teachers_votes.shape[0]
    nb_classes = teachers_votes.shape[1]
    labels_ln_max = np.zeros(nb_samples, dtype=int)

    for i in range(nb_samples):
        labels_ln_max[i] = np.argmax( teachers_votes[i] + np.random.laplace(0, gamma, nb_classes) )

    return labels_ln_max


def aggregate_gn_max(teachers_votes, sigma):

    nb_samples = teachers_votes.shape[0]
    nb_classes = teachers_votes.shape[1]
    labels_gn_max = np.zeros(nb_samples, dtype=int)

    for i in range(nb_samples):
        labels_gn_max[i] = np.argmax( teachers_votes[i] + np.random.normal(0, sigma, nb_classes) )

    return labels_gn_max


def aggregate_cgn_max(teachers_votes, threshold, sigma1, sigma2):

    nb_samples = teachers_votes.shape[0]
    nb_classes = teachers_votes.shape[1]
    labels_cgn_max = [] # We can't initialize a numpy array as we don't know yet its length
    indices_answered = []

    for i in range(nb_samples):
        if ( np.max(teachers_votes[i]) + np.random.normal(0, sigma1) >= threshold ):
            labels_cgn_max.append(  np.argmax( teachers_votes[i] + np.random.normal(0, sigma2, nb_classes) )  )
            indices_answered.append(i)

    return (np.array(labels_cgn_max), np.array(indices_answered))



#######################################################################



if __name__ == "__main__":

    # Will quit the program if incorrect options are passed
    # to the script
    parse_params()

    # Get the 3 common options to all mechanism
    aggregation_mechanism = sys.argv[1]
    input_preds_folder_path = sys.argv[2]
    output_numpy_file_path = sys.argv[3]

    # Get all teachers predictions in a big numpy array (nb_teachers x nb_samples)
    nb_teachers, nb_samples, nb_classes, teachers_predictions = collect_teachers_predictions(input_preds_folder_path)

    # Get teachers votes in a big numpy array (nb_samples x nb_classes)
    teachers_votes = compile_teachers_votes_from_predictions(nb_teachers, nb_samples, nb_classes, teachers_predictions)

    # Logic to select the right aggregation mechanism
    if aggregation_mechanism == "no_noise":
        student_labels = aggregate_no_noise(teachers_votes)

    if aggregation_mechanism == "ln_max":
        gamma = int(sys.argv[4])
        student_labels = aggregate_ln_max(teachers_votes, gamma)

    if aggregation_mechanism == "gn_max":
        sigma = int(sys.argv[4])
        student_labels = aggregate_gn_max(teachers_votes, sigma)

    if aggregation_mechanism == "cgn_max":
        threshold = int(sys.argv[4])
        sigma1 = int(sys.argv[5])
        sigma2 = int(sys.argv[6])
        student_labels, indices_answered = aggregate_cgn_max(teachers_votes, threshold, sigma1, sigma2)

    # Store the student labels at the path indicated by user
    np.save(output_numpy_file_path, student_labels)

    # In the case of Confident GN-MAX, we also store the array of answered queries indices:
    if aggregation_mechanism == "cgn_max":
        np.save("answered_indices_" + output_numpy_file_path, indices_answered)

    # Print closing message
    print("["+bcolors.GREEN+"OK"+bcolors.ENDC+"] Student Labels succesfully saved as %s" % (output_numpy_file_path))
    exit(0)

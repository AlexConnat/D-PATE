import sys
import os
import numpy as np


class bcolors:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'


#########################################################################


# A little help text of the aggregator module
def usage():

    print("")
    print(bcolors.UNDERLINE + "Usage:" + bcolors.ENDC + " ./%s <aggregation mechanism> <input folder> <output path> [<parameter(s)>]" % (sys.argv[0]))
    print("")
    print(bcolors.BOLD + "Options:" + bcolors.ENDC)
    print("- aggregation mechanism: choose from 'no_noise', 'ln_max', 'gn_max', and 'cgn_max'")
    print("- input folder: path of the folder where the teachers predictions are")
    print("- output path: name of the file(s) containing the aggregated predictions should have")
    print("- parameters: depends on the aggregation mechanism selected (see under for examples)")
    print("")
    print(bcolors.BOLD + "Examples:" + bcolors.ENDC)
    print("- ./%s no_noise predictions_folder agg_no_noise.npy                               (no params)" % (sys.argv[0]))
    print("- ./%s ln_max predictions_folder agg_ln_max.npy <laplacian scale>                 (e.g: 20)" % (sys.argv[0]))
    print("- ./%s gn_max predictions_folder agg_gn_max.npy <sigma>                           (e.g: 100)" % (sys.argv[0]))
    print("- ./%s cgn_max predictions_folder agg_cgn_max.npy <threshold> <sigma1> <sigma2>   (e.g: 300 200 40)" % (sys.argv[0]))
    print("")


# Doesn't return anything, but will quit the program if some options or parameters are incorrect
def parse_params():

 # Didn't provide aggregation mechanism
 if len(sys.argv) < 2:
     print(bcolors.RED + "Error:" + bcolors.ENDC + " Please provide an aggregation mechanism.")
     usage()
     exit(-1)

 # Didn't provide paths for input folder or output file
 if len(sys.argv) < 4:
     print(bcolors.RED + "Error:" + bcolors.ENDC + " Please provide both an input folder path, and an output file path.")
     usage()
     exit(-1)

 # Check the whether the path of the predictions folder is valid
 input_preds_folder_path = sys.argv[2]
 if not os.path.isdir(input_preds_folder_path):
     print(bcolors.RED + "Error:" + bcolors.ENDC + " Please provide a valid input folder (predictions) path.")
     usage()
     exit(-1)

 # Check the whether the path of the output numpy file is valid
 output_numpy_file_path = sys.argv[3]
 try:
     np.save(output_numpy_file_path, np.zeros(1)) # To test it, we create an (almost) empty file (contains only 1 zero)
 except FileNotFoundError:
     print(bcolors.RED + "Error:" + bcolors.ENDC + " Please provide a valid output file (.npy file) path.")
     usage()
     exit(-1)

 # Check whether the first argument is among the 4 valid aggregation mechanisms names
 if (sys.argv[1] not in ["no_noise", "ln_max", "gn_max", "cgn_max"]):
     print(bcolors.RED + "Error:" + bcolors.ENDC + " Please provide a valid aggregation mechanism.")
     usage()
     exit(-1)

 # Check whether the laplacian scale parameter is here and is an integer
 if sys.argv[1] == "ln_max":
     try:
         gamma = sys.argv[4]
     except IndexError:
         print(bcolors.RED + "Error:" + bcolors.ENDC + " Please provide a laplacian scale parameter.")
         usage(); exit(-1)

     try:
         gamma = int(sys.argv[4])
     except ValueError:
         print(bcolors.RED + "Error:" + bcolors.ENDC + " The laplacian scale parameter should be an integer value.")
         usage(); exit(-1)

 # Check whether the sigma parameter is here and is an integer
 if sys.argv[1] == "gn_max":
     try:
         sigma = sys.argv[4]
     except IndexError:
         print(bcolors.RED + "Error:" + bcolors.ENDC + " Please provide a sigma std. deviation parameter.")
         usage(); exit(-1)

     try:
         sigma = int(sys.argv[4])
     except ValueError:
         print(bcolors.RED + "Error:" + bcolors.ENDC + " The sigma std. deviation parameter should be an integer value.")
         usage(); exit(-1)

 # Check whether the laplacian scale parameter is here and is an integer
 if sys.argv[1] == "cgn_max":
     try:
         T = sys.argv[4]
         sigma1 = sys.argv[5]
         sigma2 = sys.argv[6]
     except IndexError:
         print(bcolors.RED + "Error:" + bcolors.ENDC + " Please provide the 3 required parameters: Threshold, Sigma1, Sigma2.")
         usage(); exit(-1)

     try:
         T = int(sys.argv[4])
         sigma1 = int(sys.argv[5])
         sigma2 = int(sys.argv[6])
     except ValueError:
         print(bcolors.RED + "Error:" + bcolors.ENDC + " The parameters should be integer values.")
         usage(); exit(-1)

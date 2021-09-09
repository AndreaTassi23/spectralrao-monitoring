import numpy as np
import pandas as pd
import math
from itertools import combinations
import rasterio
from rasterio import plot
import matplotlib.pyplot as plt


def spectralrao(data_input, output_path, distance_m="euclidean", p=np.nan, window=9, mode="classic", na_tolerance=0.0,
                simplify=1, debugging=False):
    def euclidean_distance(pair_list):
        #         Compute euclidean distance between two vectors
        tmp = [(x[0] - x[1]) ** 2 for x in pair_list]

        return math.sqrt(sum(tmp))

    def mana_distance(pair_list):
        # compute manatthan distance between two vectors
        tmp = [abs(x[0] - x[1]) for x in pair_list]
        return sum(tmp)

    def compute_isNaN(inputArr, valueOfNan=1):
        # Return an array with valueOFNan instead of the NaN of the input array
        inputArr = inputArr.ravel()
        outArr = np.zeros(len(inputArr))
        for i in range(len(inputArr)):
            if inputArr[i] != valueOfNan:
                outArr[i] = 1
        return outArr

    def tiff_to_numpy(tiff_input):
        # Convert Tiff input tu Numpy
        matrix1 = tiff_input.read()
        matrix1 = matrix1.reshape((matrix1.shape[1]), matrix1.shape[2])
        # print(matrix1)
        minNum = -999
        matrix1[matrix1 == minNum] = np.nan
        return matrix1

    def export_tiff(naip_meta, output_rao, output_path):
        # Write the computation output on Tiff file
        naip_transform = naip_meta["transform"]
        naip_crs = naip_meta["crs"]
        naip_transform, naip_crs
        naip_meta['count'] = 1
        naip_meta['dtype'] = "float64"
        with rasterio.open(output_path, 'w', **naip_meta) as dst:
            dst.write(output_rao, 1)

    integer_dtypes = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
    float_dtypes = ("float_", "float16", "float32", "float64")

    if mode == "classic":  # one dimension input
        info = data_input.profile
        data_input = tiff_to_numpy(data_input)
        rasterm = np.copy(data_input)

        isfloat = False  # If data are float numbers, transform them in integer
        if (rasterm.dtype.name not in integer_dtypes):
            print("Converting input data in an integer matrix...")
            isfloat = True
            mfactor = np.power(1000000, simplify)
            rasterm = rasterm * mfactor
            rasterm = rasterm.astype('int64')
        #           print(rasterm)

        else:
            rasterm = rasterm.astype('int64')
        #   rasterm = np.transpose(rasterm)
    if window % 2 == 1:
        w = int((window - 1) / 2)
    else:
        raise Exception("The size of moving window must be an odd number. Exiting...")

    # Rao_Q initialization to NaN
    if type(data_input) is list:
        print(True)
        raoqe = np.zeros(shape=data_input[0].shape)
    else:
        raoqe = np.zeros(shape=rasterm.shape)
    raoqe[:] = np.nan

    # Check if there are NaN value which becomes int
    nan_to_int64 = np.array([np.nan]).astype('int64')[0]

    if mode == 'classic':  # one dimension input
        intNaNPresence = nan_to_int64 in rasterm 
        # Reshape values
        s = pd.Series(rasterm.flatten('F'), dtype="category")
        values = [s.cat.codes[i] + 1 for i in range(len(s.cat.codes))]
        # print(values)
        rasterm_1 = np.array(
            [[values[x + y * rasterm.shape[0]] for y in range(rasterm.shape[1])] for x in range(rasterm.shape[0])])

        # Add fake columns and rows for moving window
        trasterm = np.zeros(shape=(rasterm.shape[0] + 2 * w, rasterm.shape[1] + 2 * w))
        trasterm[:] = np.nan
        trasterm[w:w + rasterm.shape[0], w:w + rasterm.shape[1]] = rasterm_1[:]
        # print(trasterm)

        # Derive distance matrix
        classes_values = np.array(s.cat.categories)
        # print("------------------------------------")
        # print(classes_values)
        nan_to_int64 = np.array([np.nan]).astype('int64')[0]

        # Loop over each pixel
        for cl in range(w, w + rasterm.shape[1]):
            for rw in range(w, w + rasterm.shape[0]):
                print('Computing pixel {:d},{:d}'.format(rw, cl))

                # Check if the pixel is on the border
                borderCondition = np.sum(
                    np.invert(np.isnan(trasterm[rw - w:rw + w + 1, cl - w:cl + w + 1]))) < np.power(window, 2) - (
                                          (np.power(window, 2)) * na_tolerance)
                # Check if there are too many nans in the window
                tooManyIntNan = np.sum(compute_isNaN(trasterm[rw - w:rw + w + 1, cl - w:cl + w + 1])) < np.power(window,
                                                                                                                 2) - (
                                        (np.power(window, 2)) * na_tolerance)
                if (intNaNPresence and tooManyIntNan) or borderCondition:
                    # It is a pixel to jump if it is on the border or if there are nans and too many in the window
                    # print('Skipping pixel {:d},{:d}'.format(rw,cl))
                    pass
                else:  # Pixel to execute                    
                    tw = pd.Series(trasterm[rw - w:rw + w + 1, cl - w:cl + w + 1].flatten('F'),
                                   dtype="category").value_counts(ascending=True)
                    tw_values = np.array(tw.values)
                    tw_labels = np.array(tw.index)

                    # to use if value_counts parameter dropna is set to False (default True)
                    if np.sum(np.isnan(np.array(tw.index))) > 0:
                        pass  # here you have to drop the na entry, but pd.Series handles it on its own

                    if debugging:
                        print("Working on coords ", rw, ",", cl, ". classes length: ", len(tw_values), ". window size=",
                              window)

                    # if clause to exclude windows with only 1 category
                    if (len(tw_values) < 2):
                        pass  # THere is only the same value in the window then skip
                    else:
                        if (nan_to_int64 in tw_labels):
                            temp_index = np.where(tw_labels == nan_to_int64)
                            p = tw_values / (np.sum(tw_values) - tw_values[index])
                            p[temp_index] = 0
                        else:
                            p = tw_values / np.sum(tw_values)
                        p1 = np.zeros(shape=(len(tw_labels), len(tw_labels)))
                        for r in range(len(tw_labels)):
                            for c in range(r + 1, len(tw_labels)):
                                p1[r, c] = p[c] * p[r]

                        # Find the indices of the classes to use from distance matrix d1 to build d2
                        indices = np.array([int(el) - 1 for el in tw_labels])

                        # Compute value in the windows for only the current pixwl
                        d2 = np.zeros(shape=(len(indices), len(indices)))
                        for i, r in enumerate(indices):
                            for j, c in enumerate(indices):
                                if ((classes_values[r] == nan_to_int64) | (
                                        classes_values[c] == nan_to_int64)):  # distanza 0 se un valore e' nan
                                    d2[i, j] = 0
                                else:
                                    d2[i, j] = classes_values[c] - classes_values[r]

                        # Finally compute RAO for the current pixel
                        if isfloat:
                            raoqe[rw - w, cl - w] = np.sum(p1 * d2) / mfactor
                        else:
                            raoqe[rw - w, cl - w] = np.sum(p1 * d2)

        # End of for loop 
        print(("\nCalculation of Rao's index complete.\n"))
        # End of classic Rao computation


    elif mode == 'multidimension':  # multi dimensions input
        info = data_input[0].profile
        # Convertion to Numpy and add to the same list
        numpy_data = []
        for inpu in data_input:
            numpy_data.append(tiff_to_numpy(inpu))
        if debugging:
            print("check: Into multidimensional clause")
        trastersm_list = []
        for mat in numpy_data:
            trasterm = np.zeros(shape=(mat.shape[0] + 2 * w, mat.shape[1] + 2 * w))
            trasterm[:] = np.nan
            trasterm[w:w + mat.shape[0], w:w + mat.shape[1]] = mat
            trastersm_list.append(trasterm)

        # Iterate over each pixel
        for cl in range(w, numpy_data[0].shape[1] + w):
            for rw in range(w, numpy_data[0].shape[0] + w):
                print('Computing pixel {:d},{:d}'.format(rw, cl))
                borderCondition = [
                    np.sum(np.invert(np.isnan(x[rw - w:rw + w + 1, cl - w:cl + w + 1]))) < np.power(window, 2) - (
                            (np.power(window, 2)) * na_tolerance) for x in trastersm_list]
                if True in borderCondition:
                    # Skipping the pixel which does not respect border condition (both NaN tollerance and distance to border)
                    pass
                else:  # Execute the RAO for the current pixel
                    tw = [x[rw - w:rw + w + 1, cl - w:cl + w + 1] for x in
                          trastersm_list]  # compute the window for each pixel
                    lv = [x.ravel() for x in tw]  # linearize the matrices

                    # Compute all the possibile point combination
                    vcomb = combinations(range(lv[0].shape[0]), 2)
                    vcomb = list(vcomb)
                    vout = []
                    # Compute value for each combination
                    for comb in vcomb:
                        lpair = [[x[comb[0]], x[comb[1]]] for x in lv]
                        if distance_m == 'euclidean':  # Distance option is Euclidean
                            out = euclidean_distance(lpair)
                        elif distance_m == 'mana':  # Distance option is manattan
                            out = man_distance(lpair)
                        vout.append(out)
                    # Rescale output
                    vout_rescaled = [x * 2 for x in vout]
                    vout_rescaled[:] = [x / window ** 4 for x in vout_rescaled]
                    # Final RAO computation for current pixel
                    raoqe[rw - w, cl - w] = np.nansum(vout_rescaled)

    output = [raoqe]
    export_tiff(info, raoqe, output_path)
    return output

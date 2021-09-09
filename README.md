The open-source repository called spectralrao-monitoring is implemented on Python3.

A function called spectralrao receives one or more rasters as input and converts each one of them to a NumPy 2d array and performs all the steps needed to retrieve the RaoQ index and saves the output as a GeoTIFF raster in the desired path.
The function supports two different modes of use: 
(i) "classic" calculates RaoQ on a single raster layer (or band)
(ii) multi-dimensional, making use of several bands defined by the user

The window size parameter given by the user is required to define an odd-sized square rolling window to preserve the integrity of the information - the movement of the window on the whole image allowing to iterate the procedures to define RaoQ.

The parameter "na.tolerance" determines the algorithm's behavior when encountering Not a Number (NaN) values in the input image. This parameter represents the minimum number of finite values accepted, enabling the calculation of the RaoQ in any given rolling window



A function called threshold_method allowed to define the threshold using the secant method. This is defined using two methods available in the Python package: calculateDistance used to calculate the distance between two points. It is mandatory to compute the threshold_method to determine the interest value to define the limit to certificate a change. This last method requires as input the NumPy type difference between the RaoQ for two periods of interest, being the results is a binary map (Change-NoChange)



The binary_change method allowed to define a single binary map where the change is confirmed only when it occurred in the same pixel of two different inputs



The method, priority_classification, requires as input the two binary maps extracted using RaoQ and NDVI information and allows to define the priority of the change for each pixels of the area of interest.

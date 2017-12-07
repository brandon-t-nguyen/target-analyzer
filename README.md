# Target Analyzer

There are multiple scripts here. The scripts have only been tested on a Linux system.

## Dependencies
* OpenCV 3
* Python 3
* Numpy

## Scripts
### target_analyzer.py
This is the prototype user interaction implementation. It takes as a parameter
the path to an image.

### dataset_analyzer.py
This is a prototype automated implementation. It uses a CSV file that
contains file names and coordinates. A sample dataset can be provided upon request.
It will run over the dataset and calculate the performance metrics. It can
also perform the hyperparameter tuning.

### gen_data.py
This is a helper script that allows the user to generate an entry for a data-set.
Provide the name of the file as a paremeter and follow the instructions as they
are printed out.

### analyzer_prototype.py (deprecated)
Original prototype implementation. Does not work well.

### clicker.py
Original version of gen_data.py. Will print out the coordinates clicked for an image.

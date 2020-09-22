# CosinorPy

CosinorPy presents a python package for cosinor based rhythmometry. It is composed of three modules:
* [```file_parser```](docs/docs_file_parser.md): reading and writting xlsx or csv files and generating synthetic data

* [```cosinor```](docs/docs_cosinor.md): single- or multi-component cosinor functions

* [```cosinor1```](docs/docs_cosinor1.md): single-component cosinor specific functions

To use these three modules include the following code in your python file:

```from CosinorPy import file_parser, cosinor, cosinor1```

CosinorPy can be used in a combination with different types of experimental data (e.g., qPCR data -- independent measurement, real-time luminescence data -- dependent measurements or even count data for which Poisson regression is used). Input data need to be formatted in accordance with the implementation of `file_parser` module (see [```file_parser```](docs/docs_file_parser.md)). This module implements several pre-processing functions that can be applied on the data, such as removal of outliers, removal of the accumulation of luminescence, removal of linear component in the measurements etc. Moreover, the user might as well preprocess the data with alternative methods, e.g., with the application of a lowpass filter. After the data has been imported, different types of analyses can be applied, which are described in the examples below. 

## Installation

CosinorPy can be installed using ```pip``` with the command

```pip install CosinorPy```

## Examples
Examples are given as interactive python notebook (ipynb) files:
* [```demo_independent.ipynb```](demo_independent.ipynb): cosinor analysis of independent data
* [```demo_dependent.ipynb```](demo_dependent.ipynb): cosinor analysis of population (dependent) data
* [```demo_csv.ipynb```](demo_csv.ipynb): reading from csv file 
* [```demo_xlsx.ipynb```](demo_xlsx.ipynb): reading from xlsx file

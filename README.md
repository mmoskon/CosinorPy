# CosinorPy

CosinorPy presents a python package for cosinor based rhythmometry. It is composed of three modules:
* [`file_parser`](docs/docs_file_parser.md): reading and writting xlsx or csv files and generating synthetic data

* [`cosinor`](docs/docs_cosinor.md): single- or multi-component cosinor functions

* [`cosinor1`](docs/docs_cosinor1.md): single-component cosinor specific functions

To use these three modules include the following code in your python file:

`from CosinorPy import file_parser, cosinor, cosinor1`

CosinorPy can be used in a combination with different types of experimental data (e.g., qPCR data - independent measurement, bioluminescence data - dependent measurements, or even count data for which Poisson regression is used). Input data need to be formatted in accordance with the implementation of the `file_parser` module (see [`file_parser`](docs/docs_file_parser.md)). This module implements several pre-processing functions that can be applied to the data, such as removal of outliers, removal of the linear component in the data, removal of the data outside a given time interval, etc. Moreover, the user might as well preprocess the data with alternative methods, e.g., with the application of a lowpass filter. When collecting the data, the user should follow the guidelines for circadian analyses as described in [1]. Moreover, before collecting the samples, the user can approximate the minimal required sample size to obtain the required accuracy [2] (see `cosinor1.amplitude_detection`, `cosinor1.amplitude_confidence` and `cosinor1.acrophase_confidence`). After the data has been imported, different types of analyses can be applied. These are described in more details in the examples below and in the paper [3].

## Installation

CosinorPy can be installed using `pip` with the command:

`pip install CosinorPy`

To install the software version described in [3], the following command should be issued:

`pip install CosinorPy==1.1`

## Examples
Examples are given as interactive python notebook (ipynb) files:
* [`demo_independent.ipynb`](demo_independent.ipynb): cosinor analysis of independent data
* [`demo_dependent.ipynb`](demo_dependent.ipynb): cosinor analysis of population (dependent) data
* [`demo_csv.ipynb`](demo_csv.ipynb): reading from csv file 
* [`demo_xlsx.ipynb`](demo_xlsx.ipynb): reading from xlsx file
* [`multi_vs_single.ipynb`](multi_vs_single.ipynb): multi-component versus single-component cosinor model

The repository as well includes the following R scripts: [`cosinor2_independent.R`](cosinor2_independent.R), [`cosinor2_independent_compare.R`](cosinor2_independent_compare.R), [`cosinor2_dependent.R`](cosinor2_dependent.R) and [`cosinor2_dependent_compare.R`](cosinor2_dependent_compare.R). These can be used to reproduce some of the results obtained with CosinorPy using cosinor and cosinor2 R packages.

## Questions
* [Why is the resolution of the periodograms so low?](https://github.com/mmoskon/CosinorPy/blob/master/docs/periodograms.md)

## How to cite CosinorPy
If you are using CosinorPy for your scientific work, please cite:

Moškon, M. "CosinorPy: A Python Package for cosinor-based Rhythmometry." BMC Bioinformatics 21.485 (2020).

The full paper is available at <https://www.doi.org/10.1186/s12859-020-03830-w>.

## Contact
Please direct your questions and comments to [miha.moskon@fri.uni-lj.si](mailto:miha.moskon@fri.uni-lj.si)

## References

[1] Hughes, Michael E., et al. "Guidelines for genome-scale analysis of biological rhythms." Journal of biological rhythms 32.5 (2017): 380-393.

[2] Bingham, Christopher, et al. "Inferential statistical methods for estimating and comparing cosinor parameters." Chronobiologia 9.4 (1982): 397-439.

[3] Moškon, M. "CosinorPy: A Python Package for cosinor-based Rhythmometry." BMC Bioinformatics 21.485 (2020).
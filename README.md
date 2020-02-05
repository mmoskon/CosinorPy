# CosinorPy

CosinorPy presents a python package for cosinor based rhythmomethry. It is composed of three modules:
* [```file_parser```](docs/docs_file_parser.md): reading and writting xlsx or csv files and generating synthetic data

* [```cosinor```](docs/docs_cosinor.md): single- or multi-component cosinor functions

* [```cosinor1```](docs/docs_cosinor1.md): single-component cosinor specific functions

## Installation

To use these three modules include the following code in your python file:

```from CosinorPy import file_parser, cosinor, cosinor1```

CosinorPy can be installed using ```pip``` with the command

```pip install CosinorPy```

## Examples
Examples are given as interactive python notebooks (ipynb):
* [```demo_independent.ipynb```](demo_independent.ipynb): cosinor analysis of independent data
* [```demo_dependent.ipynb```](demo_dependent.ipynb): cosinor analysis of population (dependent) data
* [```demo_csv.ipynb```](demo_csv.ipynb): reading from csv file 
* [```demo_xlsx.ipynb```](demo_xlsx.ipynb): reading from xlsx file

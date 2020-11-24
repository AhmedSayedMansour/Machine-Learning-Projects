# [Coursera Machine Learning MOOC by Andrew Ng](https://www.coursera.org/learn/machine-learning) 
# Python Programming Assignments

![](machinelearning.jpg)

This repositry contains my solution for the python versions of the programming assignments for the [Machine Learning online class](https://www.coursera.org/learn/machine-learning) taught by Professor Andrew Ng. This is perhaps the most popular introductory online machine learning class. In addition to being popular, it is also one of the best Machine learning classes any interested student can take to get started with machine learning. An unfortunate aspect of this class is that the programming assignments are in MATLAB or OCTAVE, probably because this class was made before python became the go-to language in machine learning.

Here are some new and useful features for these sets of assignments: 

- The assignments use [Jupyter Notebook](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html), which provides an intuitive flow easier than the original MATLAB/OCTAVE assignments.
- The original assignment instructions have been completely re-written and the parts which used to reference MATLAB/OCTAVE functionality have been changed to reference its `python` counterpart.
- The re-written instructions are now embedded within the Jupyter Notebook along with the `python` starter code. For each assignment, all work is done solely within the notebook.
- The `python` assignments can be submitted for grading. They were tested to work perfectly well with the original Coursera grader that is currently used to grade the MATLAB/OCTAVE versions of the assignments. 
- After each part of a given assignment, the Jupyter Notebook contains a cell which prompts the user for submitting the current part of the assignment for grading.  

## Downloading the original version of the Assignments

To get started, you can start by either downloading a zip file of these assignments by clicking on the `Clone or download` button. If you have `git` installed on your system, you can clone this repository using : 

    clone https://github.com/dibgerge/ml-coursera-python-assignments.git
    
Each assignment is contained in a separate folder. For example, assignment 1 is contained within the folder `Exercise1`. Each folder contains two files: 
 - The assignment `jupyter` notebook, which has a `.ipynb` extension. All the code which you need to write will be written within this notebook.
 - A python module `utils.py` which contains some helper functions needed for the assignment. Functions within the `utils` module are called from the python notebook. You do not need to modify or add any code to this file.

## Requirements 

These assignments has been tested and developed using the following libraries: 

    - python==3.6.4
    - numpy==1.13.3
    - scipy==1.0.0
    - matplotlib==2.1.2
    - jupyter==1.0.0
    - jupyter-client==5.0.1
    
I recommend using at least these versions of the required libraries or later. Python 2 is not supported. 

## Caveats and tips

- In many of the exercises, the regularization parameter $\lambda$ is denoted as the variable name `lambda_`, notice the underscore at the end of the name. This is because `lambda` is a reserved python keyword, and should never be used as a variable name.

-  In `numpy`, the function `dot` is used to perform matrix multiplication. The operation '*' only does element-by-element multiplication (unlike MATLAB). If you are using python version 3.5+, the operator '@' is the new matrix multiplication, and it is equivalent to the `dot` function.

## Acknowledgements

- Some of the material used, especially the code for submitting assignments for grading is based on [`mstampfer`'s](https://github.com/mstampfer/Coursera-Stanford-ML-Python) python implementation of the assignments. 

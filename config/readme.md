# Configuration Instructions
This file outlines the instructions for creating a virtual environment and installing all the project dependencies.
These instructions assume the use of conda and pip and also assume they are both already installed and configuired as necessary.
The packages used within all aspects of this project are outlined in the requirements.txt file.
Use of other virtual environment suites is also possible by using equivelant commands as long as requirements.txt is also installed.

## Initial Creation & Activation of Virtual Environment
Using a terminal enter the following two commands to first create a python environment, and secondly to activate/enter this environment:<br>
conda create -n env python=3.11
conda activate env

## Installation of Project Dependencies/Packages
Within the terminal and the activated environment (named 'env'), run the following command to install the project dependencies into the virtual environment.

pip install -r requirements.txt

Note: for this command to work your terminal location must be within the config dir, otherwise run the below commmand, replacing the 'path/to/' with the actual path to requirements.txt from your current location.

pip install -r /path/to/requirements.txt



#!/bin/bash

# Create or update the conda environment
conda env create -f environment.yml || conda env update -f environment.yml

# Activate the environment
source activate youtube_assistant

# Install packages using pip
python3 -m pip install openai
pip3 install python-dotenv
pip3 install langchain
pip3 install --upgrade langchain
pip3 install docarray
pip3 install tiktoken


# ------------------------ setup.sh --------------------------
# How to run bash script in Terminal

#1. Open the terminal.
#2. Navigate to the directory containing the setup.sh file.
#3. Run the following command:
# chmod +x setup.sh
#This command makes the setup.sh file executable.
#4. Run the script with:
# ./setup.sh
# Project Setup Guide

This guide will help you set up the development environment for this project.

## Prerequisites

- Python 3.8 or higher
- Git
- Either Conda or Python venv (instructions for both provided below)

## Setup Instructions

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/DeepThinkers-Rrsearch/automata_convertor.git
cd repository-name
```

### 2. Create Virtual Environment

You can use either Conda or Python's built-in venv. Choose one of the options below:

#### Option A: Using Conda

```bash
# Create a new conda environment
conda create --name /myenv python=3.10

# Activate the environment
conda activate /myenv
```

#### Option B: Using Python venv

```bash
# Create a virtual environment
python -m venv myenv

# Activate the environment
# On Windows:
myenv\Scripts\activate

# On macOS/Linux:
source myenv/bin/activate
```

### 3. Environment Variables (.env file)

An `.env` file is not required for now, but may be needed for future configurations such as API keys or database connections.

### 4. Install Required Packages

After activating your virtual environment, install all required dependencies using the requirements.txt file:

```bash
pip install -r requirements.txt
```

This will install the following packages:

- streamlit
- torch
- numpy
- google-generativeai
- matplotlib
- pandas

## Running the Application

After completing the setup, you can run the application using:

```bash
streamlit run app.py
```

## Deactivating the Environment

When you're done working on the project, you can deactivate the virtual environment:

```bash
# For both conda and venv
deactivate
```

## Troubleshooting

- If you encounter any package installation errors, try upgrading pip first: `pip install --upgrade pip`
- For PyTorch installation issues, visit the [official PyTorch website](https://pytorch.org/) for platform-specific installation instructions
- Make sure your Python version is compatible with all the required packages

## Additional Notes

- Always activate your virtual environment before working on the project
- Keep the `requirements.txt` file updated if you add new dependencies
- Consider using `pip freeze > requirements.txt` to update the requirements file with exact versions

## Chat Model Tutorial Link

- https://python.langchain.com/docs/tutorials/chatbot/

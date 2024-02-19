# Seph's MVP

## Overview
This project involves a complex machine learning model integrating an RBM Layer, a Q-Learning Layer, and Transformer Encoder. It is designed for experimenting with reinforcement learning techniques and neural network architectures. Additionally, it includes a model evaluation script (`model_eval.py`) for assessing the performance of the trained model.

![Seph's Biome](https://github.com/LoQiseaking69/SephsBIOME/blob/master/Docs/Misc/IMG_7130.png)

## Contents
- `model.py`: The main script containing the model's architecture and training logic. It includes the RBM Layer, ReplayBuffer, QLearningLayer, and other necessary components for model building.
- `model_eval.py`: Script for evaluating the trained model on the environment over a specified number of episodes.
- `requirements.txt`: Lists all the Python packages required to run the project.
- `README.md`: This file, providing a comprehensive guide to the project.
- `main.yml`: Configuration for the GitHub Actions CI/CD pipeline.

## Setup and Installation
To set up the project:
1. Ensure Python 3.8+ is installed.
2. Clone the repository to your local machine.
3. Install dependencies: Run `pip install -r requirements.txt` in your project directory.
4. Run the script: Execute `python model.py` to start the model training or testing.

## Usage
- The `model.py` script can be run directly after setting up the environment. It will initiate the model building process and output the results.
- The `model_eval.py` script evaluates the trained model's performance on the environment over a specified number of episodes. To use it, execute `python model_eval.py` after training the model.
- To modify the model, edit the `model.py` script, adjusting layers, parameters, or the training process as needed.

## Contributing
Contributions to this project are welcome. Feel free to fork the repository, make changes, and submit pull requests. Please document your changes and additions clearly.

## License
This project is open-source and available under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html). For more details, see the LICENSE file in the repository.

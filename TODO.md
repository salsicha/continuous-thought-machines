To create a new RL task for Super Mario Bros. in this project, follow these steps:

Environment Setup:

Choose a Python library for Mario emulation and RL (e.g., gym-super-mario-bros, nes-py, or retro).
Install the required packages (e.g., gym-super-mario-bros, stable-baselines3, torch).
Create a New Task Directory:

In rl, create a new subdirectory, e.g., super_mario/.
Add Training Scripts:

In tasks/rl/super_mario/, add training scripts (e.g., train_ctm_mario.sh, train_lstm_mario.sh) similar to other environments.
Create a Python training script (e.g., train.py) that:
Initializes the Mario environment.
Loads your RL model (CTM, LSTM, etc.).
Trains the agent to play Mario.
Add Analysis and Plotting:

Add an analysis/ folder for evaluation scripts and plotting results.
Implement scripts to visualize performance (e.g., reward curves, gameplay videos).
Update Utilities:

If needed, add or modify utility functions in utils.py to support the new environment.
Add Bash Scripts:

In tasks/rl/super_mario/scripts/, add shell scripts to automate training and evaluation, following the pattern in other tasks.
Documentation:

Add a README.md in tasks/rl/super_mario/ explaining how to train and evaluate the Mario agent.
Testing:

Optionally, add tests in tests to validate the new environment and training loop.
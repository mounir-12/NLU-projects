# Models

# Baseline RNN
A simple bidirectional RNN that takes the five sentences concatenated with spaces and classify to 0 or 1 based on if it's the true continuation.
Doesn't seem to be able to learn anything
# BERT
https://arxiv.org/abs/1810.04805

Idea: Use the first four sentences as "sentence a" and the last as "sentence b", do binary classification on this input

Libraries:
- pandas
- tensorflow
- tensorflow hub: pip install tensorflow_hub
- Bert tensorflow: pip install bert-tensorflow
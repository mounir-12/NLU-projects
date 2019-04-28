# Natural Language Understanding 2019
## Project 1: Language Modeling with Recurrent Neural Networks and Sentence Continuation

### Authors:
 - Srikanth Sarma      sgurram@student.ethz.ch
 - Mounir Amrani       mamrani@student.ethz.ch
 - Ali Hosseiny        aliho@student.ethz.ch


 Requirements:
  - All data (`sentences.train`, `sentences.test`, `sentences.continuation`) named exactly as mentioned, should be in `./data` folder.
  - All the python scripts (`main.py`, `lang.py`, `model.py` and `load_embedding.py`) should be in the `./` directory.


**Training:**

  To train model x (1a, 1b, 1c, 2), use the command:

  `python main.py -t x`


After the model has been trained over the sentences `./data/sentences.test`, perplexity values for sentences `./data/sentences.test` will be saved at `./group17.preplexityX`. The trained model is saved at `./models/modelX.ckpt`

**Sentence Continuation**
  To perform sentence continuation, there must be a trained model `./models/modelC.ckpt` available. To perform sentence continuation for prompts `sentencces.continuation`, use the command:

  `python main.py -t 2`

  After the continuations have been computed, the full sentences can be found at `./group17.continuation`
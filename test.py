from lang import Lang

import tensorflow as tf

print(tf.__version__)

l = Lang()

print(l.pad('Hi my name is', to_length=7))

print(Lang().index2token)
print(Lang().n_tokens)

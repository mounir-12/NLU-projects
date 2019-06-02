import os
import sys

import bert
import pandas as pd
import tensorflow as tf
from bert import run_classifier

# --------------------------------------------------Data---------------------------------------
from process import concat_sentences

# tf.logging.set_verbosity(tf.logging.ERROR)

train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/val.csv')
test = pd.read_csv('data/test.csv')

sys.stdout.flush()

BERT_PATH = os.path.join('data', 'models', 'bert', 'uncased_L-12_H-768_A-12')
INIT_CHECKPOINT = os.path.join(BERT_PATH, 'bert_model.ckpt')


# ------------------------------------BERT--------------------------------------------
def create_tokenizer():
    """Get the vocab file from disk."""
    return bert.tokenization.FullTokenizer(vocab_file=os.path.join(BERT_PATH, 'vocab.txt'), do_lower_case=True)


# tokenizer = create_tokenizer_from_hub_module()
tokenizer = create_tokenizer()

print("Data manipulation ...")
sys.stdout.flush()

INPUT_SENTENCES = 4  # The sentences used as text_a
TRAIN_WITH_VAL = True

train_a = concat_sentences(train, INPUT_SENTENCES)
train_b = concat_sentences(train, 5)
y_train_bert = 1 - train.label  # For Bert 0 means true continuation and 1 means fake
train_examples = [bert.run_classifier.InputExample(None, text_a=text_a, text_b=text_b, label=label) for
                  text_a, text_b, label in zip(train_a, train_b, y_train_bert)]

val_a = concat_sentences(val, INPUT_SENTENCES)
val_b = concat_sentences(val, 5)
y_val_bert = 1 - val.label  # For Bert 0 means true continuation and 1 means fake
val_examples = [bert.run_classifier.InputExample(None, text_a=text_a, text_b=text_b, label=label) for
                text_a, text_b, label in zip(val_a, val_b, y_val_bert)]

if TRAIN_WITH_VAL:
    train_examples.extend(val_examples)

test_a = concat_sentences(test, INPUT_SENTENCES)
test_b = concat_sentences(test, 5)
y_test_bert = 1 - test.label  # For Bert 0 means true continuation and 1 means fake
test_examples = [bert.run_classifier.InputExample(None, text_a=text_a, text_b=text_b, label=label) for
                 text_a, text_b, label in zip(test_a, test_b, y_test_bert)]

label_list = [0, 1]
# We'll set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = 128

# Convert our train and test features to InputFeatures that BERT understands.
train_features = bert.run_classifier.convert_examples_to_features(train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
val_features = bert.run_classifier.convert_examples_to_features(val_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = bert.run_classifier.convert_examples_to_features(test_examples, label_list, MAX_SEQ_LENGTH, tokenizer)


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
    """Creates a classification model."""
    bert_config = bert.modeling.BertConfig.from_json_file(os.path.join(BERT_PATH, 'bert_config.json'))
    model = bert.modeling.BertModel(
        config=bert_config,
        is_training=not is_predicting,  # TODO Check this
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=True)  # TODO Check this

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    # Create our own layer to tune for politeness data.
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
            return (predicted_labels, log_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)


# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        print("\nEntering function with mode {}:".format(mode))
        sys.stdout.flush()
        # TRAIN and EVAL
        if not is_predicting:
            print("Not Predicting ...")
            sys.stdout.flush()

            (loss, predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = bert.modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                            INIT_CHECKPOINT)
            tf.train.init_from_checkpoint(INIT_CHECKPOINT, assignment_map)

            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            # Calculate evaluation metrics. 
            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_score(
                    label_ids,
                    predicted_labels)
                auc = tf.metrics.auc(
                    label_ids,
                    predicted_labels)
                recall = tf.metrics.recall(
                    label_ids,
                    predicted_labels)
                precision = tf.metrics.precision(
                    label_ids,
                    predicted_labels)
                true_pos = tf.metrics.true_positives(
                    label_ids,
                    predicted_labels)
                true_neg = tf.metrics.true_negatives(
                    label_ids,
                    predicted_labels)
                false_pos = tf.metrics.false_positives(
                    label_ids,
                    predicted_labels)
                false_neg = tf.metrics.false_negatives(
                    label_ids,
                    predicted_labels)
                return {
                    "eval_accuracy": accuracy,
                    "f1_score": f1_score,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                print("Training ...")
                sys.stdout.flush()
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  train_op=train_op)
            else:
                print("Evaluating ...")
                sys.stdout.flush()
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  eval_metric_ops=eval_metrics)
        else:
            print("Predicting ...")
            sys.stdout.flush()
            (predicted_labels, log_probs) = create_model(
                is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = bert.modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                            INIT_CHECKPOINT)
            tf.train.init_from_checkpoint(INIT_CHECKPOINT, assignment_map)

            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn


# Compute train and warmup steps from batch size
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

# Compute # train and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# Specify outpit directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir='temp/model',
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

model_fn = model_fn_builder(
    num_labels=len(label_list),
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    params={"batch_size": BATCH_SIZE})

# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = bert.run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False)

print('Beginning Training!')
sys.stdout.flush()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

print('Beginning Evaluation!')
sys.stdout.flush()
val_input_fn = run_classifier.input_fn_builder(
    features=val_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

estimator.evaluate(input_fn=val_input_fn, steps=None)

print('Beginning testing!')
sys.stdout.flush()
test_input_fn = run_classifier.input_fn_builder(
    features=test_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

estimator.evaluate(input_fn=test_input_fn, steps=None)

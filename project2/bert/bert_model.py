import os
import random
import sys
from argparse import ArgumentParser

import bert
import numpy as np
import pandas as pd
import tensorflow as tf
from bert import run_classifier
# --------------------------------------------------Data---------------------------------------
from process import concat_sentences, accuracy, prediction_choice

parser = ArgumentParser()

parser.add_argument('-trf', '--train-file', type=str, default=os.path.join('data', 'val.csv'),
                    help='Path (file) containing training data.')
parser.add_argument('-vf', '--val-file', type=str, default=os.path.join('data', 'test.csv'),
                    help='Path (file) containing evaluation data.')
parser.add_argument('-tsf', '--test-file', type=str, default=os.path.join('data', 'test.csv'),
                    help='Path (file) containing labeled test data on which prediction will be run.')
parser.add_argument('-pf', '--predict-file', type=str, default=os.path.join('data', 'predict.csv'),
                    help='Path (file) containing unlabeled data on which prediction will be run.')

parser.add_argument('-bp', '--bert-path', type=str,
                    default=os.path.join('data', 'models', 'bert', 'uncased_L-12_H-768_A-12'),
                    help='Path (directory) containing Bert model. Should contain at least the vocab and config files.')
parser.add_argument('-cp', '--checkpoint-path', type=str,
                    default=os.path.join('data', 'pretraining50k_output', 'model.ckpt-50000'),
                    help='TF checkpoint. Can be a pretrained or finetuned Bert model.')
parser.add_argument('-op', '--output-path', type=str, required=True,
                    help='Path of output files.')

parser.add_argument("-bs", "--batch-size", type=int, default=16,
                    help="Train batch size")
parser.add_argument("-e", "--epochs", type=int, default=20,
                    help="Training epochs")

parser.add_argument('-m', "--modes", nargs="+", choices=['train', 'eval', 'test', 'predict'],
                    default=['train', 'eval', 'test', 'predict'])

parser.add_argument("-rs", "--random-seed", type=int, default=9,
                    help="Random seed")

args = parser.parse_args()


modes = args.modes

random.seed(args.random_seed)
tf.set_random_seed(args.random_seed)
np.random.seed(args.random_seed)

os.makedirs(args.output_path, exist_ok=True)

BERT_PATH = args.bert_path
# INIT_CHECKPOINT = os.path.join(BERT_PATH, 'bert_model.ckpt')
INIT_CHECKPOINT = args.checkpoint_path

# tf.logging.set_verbosity(tf.logging.ERROR)

label_list = [0, 1]  # 0 means the fifth sentence is the true continuation, 1 means it isn't
MAX_SEQ_LENGTH = 128  # We'll set sequences to be at most 128 tokens (word pieces) long.

INPUT_SENTENCES = 4  # The sentences used as text_a

print("Data manipulation ...")

tokenizer = bert.tokenization.FullTokenizer(vocab_file=os.path.join(BERT_PATH, 'vocab.txt'), do_lower_case=True)


def bert_data(df):
    ids = df.InputStoryid
    texts_a = concat_sentences(df, INPUT_SENTENCES)
    texts_b = concat_sentences(df, 5)

    if 'label' in df:
        y_bert = 1 - df.label  # For Bert 0 means true continuation and 1 means fake
    else:
        y_bert = None

    if 'label' in df:
        examples = [bert.run_classifier.InputExample(id, text_a=text_a, text_b=text_b, label=label) for
                    id, text_a, text_b, label in zip(ids, texts_a, texts_b, y_bert)]
    else:
        examples = [bert.run_classifier.InputExample(id, text_a=text_a, text_b=text_b) for
                    id, text_a, text_b in zip(ids, texts_a, texts_b)]
    # Convert our examples to InputFeatures that BERT understands.
    features = bert.run_classifier.convert_examples_to_features(examples, label_list, MAX_SEQ_LENGTH, tokenizer)

    return examples, features


if args.val_file and 'eval' in modes:
    print("Reading val file:", args.val_file)
    val = pd.read_csv(args.val_file)
    val_examples, val_features = bert_data(val)
if args.train_file and 'train' in modes:
    print("Reading train file:", args.train_file)
    train = pd.read_csv(args.train_file)
    train_examples, train_features = bert_data(train)
if args.test_file and 'test' in modes:
    print("Reading test file:", args.test_file)
    test = pd.read_csv(args.test_file)
    test_examples, test_features = bert_data(test)
if args.predict_file and 'predict' in modes:
    print("Reading predictions file:", args.predict_file)
    pred = pd.read_csv(args.predict_file)
    pred_examples, pred_features = bert_data(pred)   


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
    """Creates a classification model."""
    bert_config = bert.modeling.BertConfig.from_json_file(os.path.join(BERT_PATH, 'bert_config.json'))
    model = bert.modeling.BertModel(
        config=bert_config,
        is_training=not is_predicting,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=True)

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
            print('Loading checkpoint:', INIT_CHECKPOINT)
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


print("Preparing model...")

# Compute train and warmup steps from batch size
BATCH_SIZE = args.batch_size
LEARNING_RATE = 2e-5
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 100

# Compute # train and warmup steps from batch size
if 'train' in modes:
    num_train_steps = int(len(train_features) / BATCH_SIZE * args.epochs)
else:
    # We aren't training so we don't care about the value
    num_train_steps = 1

num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# Specify output directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir=args.output_path,
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

if 'train' in modes:
    print('Beginning Training!')

    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)

    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

if 'eval' in modes:
    print('Beginning Evaluation!')

    val_input_fn = run_classifier.input_fn_builder(
        features=val_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

    print("Eval results:\n", estimator.evaluate(input_fn=val_input_fn, steps=None))

if 'test' in modes:
    print('Beginning Testing!')

    test_input_fn = run_classifier.input_fn_builder(
        features=test_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

    predictions = list(estimator.predict(input_fn=test_input_fn))
    preds = [(id, label, prediction['probabilities'][0], prediction['probabilities'][1]) for id, label, prediction in
             zip(test.InputStoryid.values, test.label.values, predictions)]
    preds_df = pd.DataFrame(preds, columns=['id', 'label', 'log_prob_true', 'log_prob_fake'])
    preds_df.to_csv(os.path.join(args.output_path, 'test_predictions.csv'), index=False)

    choice_acc, class_acc = accuracy(preds_df)
    print("Test choice accuracy: {}, test classification accuracy: {}".format(choice_acc, class_acc))

if 'predict' in modes:
    print('Beginning prediction!')

    pred_input_fn = run_classifier.input_fn_builder(
        features=pred_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

    predictions = list(estimator.predict(input_fn=pred_input_fn))
    preds = [(id, prediction['probabilities'][0], prediction['probabilities'][1]) for id, prediction in
             zip(pred.InputStoryid.values, predictions)]
    preds_df = pd.DataFrame(preds, columns=['id', 'log_prob_true', 'log_prob_fake'])
    choices = prediction_choice(preds_df)

    with open(os.path.join('data', 'predictions.csv'), 'wt') as f:
        f.write('\n'.join(str(choice) for choice in choices))

sys.stdout.flush()

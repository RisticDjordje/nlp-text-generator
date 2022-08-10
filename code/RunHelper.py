import numpy as np
import os
import random
import sys
import torch

from sklearn.metrics import classification_report, confusion_matrix


def init_random_seeds(seed_val):
    """
    Sets all random seeds

    Args:
        seed_val (int): a defined seed value
    """
    # set all random seeds
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    np.set_printoptions(threshold=sys.maxsize)


def evaluate(sentiment_analysis_trainer, device, batch_size, split):
    """
    Evaluates model on the chosen split

    Args:
        sentiment_analysis_trainer (Trainer): a trainer class that encapsulates model training
        device (str): the device where the data tensors should be stored: "cpu" or "cuda"
        batch_size (int): the batch size that should be used
        split (str): one of "train", "validation" or "test"
    """
    loss, accuracy = sentiment_analysis_trainer.evaluate(split=split, device=device, batch_size=batch_size)

    print("Running Loss: {:.3f}".format(loss))
    print("Running Accuracy: {:.3f}".format(accuracy))


def evaluate_model(sentiment_analysis_trainer, device, batch_size):
    """
    Evaluates model on the training, validation and test split

    Args:
        sentiment_analysis_trainer (Trainer): a trainer class that encapsulates model training
        device (str): the device where the data tensors should be stored: "cpu" or "cuda"
        batch_size (int): the batch size that should be used
    """
    print("Training Set")
    evaluate(sentiment_analysis_trainer, device, batch_size, split="train")
    print("\n")

    print("Validation Set")
    evaluate(sentiment_analysis_trainer, device, batch_size, split="validation")
    print("\n")

    print("Test Set")
    evaluate(sentiment_analysis_trainer, device, batch_size, split="test")


def predict(text, model, vectorizer):
    """
    Predict the sentiment of the tweet

    Args:
        text (str): the text of the tweet
        model (SentimentClassifierPerceptron): the trained model
        vectorizer (TwitterVectorizer): the corresponding vectorizer
    Returns:
        sentiment of the tweet (int), probability of that prediction (float)
    """
    # vectorize the text of the tweet
    vectorized_text = vectorizer.vectorize(text)

    # make a tensor with expected size (1, )
    vectorized_text = torch.Tensor(vectorized_text).view(1, -1)

    # run the model on the vectorized text and apply softmax activation function on the outputs
    result = model(vectorized_text, apply_softmax=True)

    # find the best class as the one with the highest probability
    probability_values, indices = result.max(dim=1)

    # take only value of the indices tensor
    index = indices.item()

    # decode the predicted target index into the sentiment, using target vocabulary
    predicted_target = vectorizer.target_vocabulary.find_index(index)

    # take only value of the probability_values tensor 
    probability_value = probability_values.item()

    return predicted_target, probability_value


def run_examples(examples, model, vectorizer):
    """
    Predict the sentiment for each tweet in the examples list

    Args:
        examples (list of str): a list of tweets
        model (SentimentClassifierPerceptron): the trained model
        vectorizer (TwitterVectorizer): the corresponding vectorizer
    """
    for text in examples:
        label, confidence = predict(text, model, vectorizer)
        print("Text: {} - Predicted label: {} - Confidence: {}".format(text, label, confidence))


def model_run_and_evaluate(dataset, vectorizer, model):
    """
    Runs model on the dataset and evaluates the predictions against the labeled data

    Args:
        dataset (TwitterDataset): a dataset used for evaluation
        vectorizer (TwitterVectorizer): the corresponding vectorizer
        model (SentimentClassifierPerceptron): the trained model
    """
    # run the model on the tweets from test sets
    y_predicted = dataset.test_df.text.apply(lambda x: predict(text=x, model=model, vectorizer=vectorizer)[0])
    # evaluate the model
    print_evaluation_report(dataset.test_df.target, y_predicted)


def print_evaluation_report(y_labeled, y_predicted):
    """
    Compares the model predictions with the labeled data

    Args:
        y_labeled (list of int): a list of labels
        y_predicted (list of int): a list of model predictions
    """
    # compare predictions with labels
    print(classification_report(y_true=y_labeled, y_pred=y_predicted))

    # plot confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix(y_true=y_labeled, y_pred=y_predicted))

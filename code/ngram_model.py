#!/usr/local/bin/python

from data_utils import utils as du
import numpy as np
import pandas as pd
import csv

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                     index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

# Load the training set
docs_train = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs_train, word_to_num)
docs_dev = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs_dev, word_to_num)


################################# DELETE THIS #############################

prob_from_bigram = 0
prob_from_trigram = 0

###########################################################################

def train_ngrams(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """
    trigram_counts = dict()
    bigram_counts = dict()
    unigram_counts = dict()
    token_count = 0
    ### YOUR CODE HERE
    def enterDic(phrase,dict):
        if phrase in dict:
            dict[phrase] += 1
        else:
            dict[phrase] = 1

    unigram_counts[word_to_num['UUUNKKK']] = 0

    for sentence in dataset:
        enterDic(sentence[1], unigram_counts) # count number of start of sentences
        enterDic((sentence[0],sentence[1]), bigram_counts) # count number of start of sentences
        for i in range(2,len(sentence)):
            if sentence[i] not in unigram_counts:
                token_count += 1
            enterDic(sentence[i], unigram_counts)
            enterDic((sentence[i-1], sentence[i]), bigram_counts)
            enterDic((sentence[i-2], sentence[i-1], sentence[i]), trigram_counts)
    ### END YOUR CODE
    return trigram_counts, bigram_counts, unigram_counts, token_count

def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    perplexity = 0
    ### YOUR CODE HERE
    def calc_prob(sentense,i, word, trigram_counts, bigram_counts, unigram_counts, train_token_count,model):
        prob = 0.0
        prev_word = sentense[i - 1]
        prev_to_prev_word = sentense[i - 2]

        if model == "unigram":
            if word in unigram_counts:
                prob = (unigram_counts[word]+0.0) / \
                       train_token_count
            else:
                prob = (unigram_counts[word_to_num['UUUNKKK']] + 0.0) / \
                       train_token_count

        if model == "bigram":
            if (prev_word,word) in bigram_counts:
              prob = (bigram_counts[(prev_word, word)] + 0.0) / \
                           unigram_counts[prev_word]
              #print(num_to_word[prev_word] ,num_to_word[word])
              #print(bigram_counts[(prev_word, word)])
              #print(unigram_counts[prev_word])
              #print("---------------------------")
            else:
                prob = 0.0

        if model == "trigram":
            if (prev_to_prev_word,prev_word,word) in trigram_counts:
                prob = (trigram_counts[(prev_to_prev_word, prev_word, word)] + 0.0) \
                         / bigram_counts[(prev_to_prev_word, prev_word)]
                        # / bigram_counts[(prev_word, word)] #this according to lecture notes slide 27
            else:
                prob = 0.0

        return prob

    l = 0
    num_of_words = 0

    for sentense in eval_dataset:
        for i,word in enumerate(sentense[2:]):
            num_of_words += 1
            prob = lambda1 * calc_prob(sentense,i+2,word, trigram_counts, bigram_counts, unigram_counts, train_token_count,"trigram") + \
                   lambda2 * calc_prob(sentense,i+2,word, trigram_counts, bigram_counts, unigram_counts, train_token_count,"bigram") +\
                   (1-lambda1-lambda2) * calc_prob(sentense,i+2,word, trigram_counts, bigram_counts, unigram_counts, train_token_count,"unigram")
            l += np.log2(prob)
    l /= num_of_words
    perplexity += 2 ** -l
    ### END YOUR CODE
    return perplexity

def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    #Some examples of functions usage
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    print "#perplexity: " + str(perplexity)
    ### YOUR CODE HERE
    print(vocabsize)
    ### END YOUR CODE

def gridSearch():
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    lambda1_range =  np.linspace(0.0, 1.0-(10**-5), num=10)
    lambda2_range = np.linspace(0.0, 1.0-(10**-5), num=10)
    combinations = [(lambda1, lambda2) for lambda1 in lambda1_range for lambda2 in lambda2_range if lambda1+lambda2<(1-10**-6)]
    print(lambda1_range)

    best_lambda1 = -1
    best_lambda2 = -1
    best_perplexity = float('inf')
    for (lambda1,lambda2) in combinations:
        perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, lambda1, lambda2)
        print("lambda1:", lambda1)
        print("lambda2:", lambda2)
        print("perplexity:", perplexity)
        print("---------------------")
        if best_perplexity > perplexity:
            best_perplexity = perplexity
            best_lambda1 = lambda1
            best_lambda2 = lambda2
    print("Best lambda1:", best_lambda1)
    print("Best lambda2:", best_lambda2)
    print("Best perplexity:", best_perplexity)


if __name__ == "__main__":
    #test_ngram()
    gridSearch()
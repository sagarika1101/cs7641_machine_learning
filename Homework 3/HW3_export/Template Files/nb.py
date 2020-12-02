import numpy as np
import json
from sklearn.feature_extraction import text


class NaiveBayes(object):

    def __init__(self):
        # load Documents
        x = open('fedpapers_split.txt').read()
        papers = json.loads(x)

        # split Documents
        papersH = papers[0]  # papers by Hamilton
        papersM = papers[1]  # papers by Madison
        papersD = papers[2]  # disputed papers

        # Number of Documents for H, M and D
        nH = len(papersH)
        nM = len(papersM)
        nD = len(papersD)

        '''To ignore certain common words in English that might skew your model, we add them to the stop words 
        list below. You may want to experiment by choosing your own list of stop words, but be sure to keep 
        'HAMILTON' and 'MADISON' in this list at a minimum, as their names appear in the text of the papers 
        and leaving them in could lead to unpredictable results '''

        stop_words = text.ENGLISH_STOP_WORDS.union({'HAMILTON', 'MADISON'})
        # stop_words = {'HAMILTON','MADISON'}
        # Form bag of words model using words used at least 10 times
        vectorizer = text.CountVectorizer(stop_words=stop_words, min_df=10)
        X = vectorizer.fit_transform(papersH + papersM + papersD).toarray()

        '''To visualize the full list of words remaining after filtering out stop words and words used less 
        than min_df times uncomment the following line'''
        # print(vectorizer.vocabulary_)

        # split word counts into separate matrices
        self.XH, self.XM, self.XD = X[:nH, :], X[nH:nH + nM, :], X[nH + nM:, :]

    def _likelihood_ratio(self, XH, XM):  # [5pts]
        '''
        Args:
            XH: nH x D where nH is the number of documents that we have for Hamilton,
                while D is the number of features (we use the word count as the feature)
            XM: nM x D where nM is the number of documents that we have for Madison,
                while D is the number of features (we use the word count as the feature)
        Return:
            fratio: 1 x D vector of the likelihood ratio of different words (Hamilton/Madison)
        '''
        # Hamilton
        total_freq_H = np.sum(XH, axis=0, keepdims=True) + 1

        # Madison
        total_freq_M = np.sum(XM, axis=0, keepdims=True) + 1

        fratio = (total_freq_H * np.sum(total_freq_M)) / (total_freq_M * np.sum(total_freq_H))
        return fratio
        # raise NotImplementedError

    def _priors_ratio(self, XH, XM):  # [5pts]
        '''
        Args:
            XH: nH x D where nH is the number of documents that we have for Hamilton,
                while D is the number of features (we use the word count as the feature)
            XM: nM x D where nM is the number of documents that we have for Madison,
                while D is the number of features (we use the word count as the feature)
        Return:
            pr: prior ratio of (Hamilton/Madison)
        '''
        words_in_H = np.sum(XH)
        words_in_M = np.sum(XM)
        prior_ratio = float(words_in_H / words_in_M)
        return prior_ratio
        # raise NotImplementedError

    def classify_disputed(self, fratio, pratio, XD):  # [5pts]
        '''
        Args:
            fratio: 1 x D vector of ratio of likelihoods of different words
            pratio: 1 x 1 number
            XD: 12 x D bag of words representation of the 12 disputed documents (D = 1307 which are the number of features for each document)
        Return:
             1 x 12 list, each entry is H to indicate Hamilton or M to indicate Madison for the corrsponding document
        '''
        res = []
        for i in range(12):
            doc = XD[i]
            t1 = np.power(fratio, doc)
            prob_of_H = np.prod(t1) * pratio
            if prob_of_H > 0.5:
                res.append('H')
            else:
                res.append('M')
        return res

        # raise NotImplementedError
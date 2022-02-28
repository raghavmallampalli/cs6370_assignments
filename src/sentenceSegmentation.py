from util import *
# Add your import statements here
import nltk
import re

class SentenceSegmentation():

    def naive(self, text):
        """
        Sentence Segmentation using a Naive Approach

        Parameters
        ----------
        arg1 : str
                A string (a bunch of sentences)

        Returns
        -------
        list
        A list of strings where each string is a single sentence
        """

        if(type_check(text)):

            # split the sentence using the '?' , '.' , '!'
            segmentedText = re.split(sent_end_chars, text)
            if '' in segmentedText: #removing unnecessary whitespaces
                segmentedText.remove('')

            return segmentedText

        else:
            print('Error!! Incorrect Datatype Input ')
            return -1

    def punkt(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : str
                A string (a bunch of sentences)

        Returns
        -------
        list
                A list of strings where each string is a single sentence
        """

        if(type_check(text)):
            #tokenizer defaults to the pre-trained version trained in English

            tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()

            tokenizer.tokenize(text)
            segmentedText = tokenizer.tokenize(text)

            return segmentedText

        else:
            print('Error!! Incorrect Datatype Input ')
            return -1
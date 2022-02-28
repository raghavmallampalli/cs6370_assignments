from util import *

# Add your import statements here
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) #creating a set with all the stopwords




class StopwordRemoval():

    def fromList(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence with stopwords removed
        """

        #Fill in code here

        if(type_check(text)):
            for sent_number,sentence in enumerate(text):#lists sentence numbers and sentences of the tokenized text

                wo_stopword_tokens = [] #tokenized sentence without stopwords initialized

                for token in sentence: #parsing through the tokens in a sentence

                    if token not in stop_words: #append to the above list if token is not a stopword

                        wo_stopword_tokens.append(token)

                text[sent_number] = wo_stopword_tokens #update the text with the stopwords removed tokenized sentence

            return text

        else:
            print('Error!! Incorrect Datatype Input ')
            return -1
from util import *

# Add your import statements here
import nltk

#for stemming
from nltk.stem.snowball import SnowballStemmer

#for lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag #part of speech

stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')



class InflectionReduction:

    #reduction using stemming
    def reduce(self, text):
        """
        Stemming/Lemmatization

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of
            stemmed/lemmatized tokens representing a sentence
        """
        reducedText = []
        #Fill in code here
        if(type_check(text)):
            for sent_number, sentence in enumerate(text): #lists sentence numbers and sentences of the tokenized text

                reduced_sent = []

                for ind,token in enumerate(sentence): #we need positional info as we're replacing with stemmed token

                    reduced_sent.append(stemmer.stem(token))


                    #text[sent_number][ind] = stemmer.stem(token) #stem the token and replace it with the unstemmed one
                reducedText.append(reduced_sent)
            return reducedText

        else:
            print('Error !! Incorrect Datatype Input')
            return -1



    #Although we might not be using lemmatization now, just here to check performance/for completeness/later use
    def reduce_lemmatizer(self, text):
        """
        Stemming/Lemmatization

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of
            stemmed/lemmatized tokens representing a sentence
        """

        #Fill in code here

        if(type_check(text)):
            for sent_number,sentence in enumerate(text): #lists sentence numbers and sentences of the tokenized text
                #first do part of speech tagging for the tokens as it'll come in handy during lemmatization
                pos_tags = pos_tag(text[sent_number])
                for ind,token in enumerate(sentence): #we need positional info as we're replacing with lemmatized token

                    #extract the tag from the list of (token,tags) (return format of pos_tag)
                    pos_token = pos_tag_to_wordnet(pos_tags[ind][1]) #pos_tag_to_wordnet is a util function
                    text[sent_number][ind] = lemmatizer.lemmatize(token,pos = pos_token)

            return text

        else:
            print('Error!! Incorrect Datatype Input ')
            return -1
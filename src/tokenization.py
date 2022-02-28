from util import *
# Add your import statements here
from nltk.tokenize.treebank import TreebankWordTokenizer
import re

class Tokenization():
    def __check_list_str(self, text):
        """
        Helper function to perform type-checking on input.
        Checks that input is a list of strings.
        """
        if not isinstance(text, list):
            raise(TypeError(
                  "Input not of list type."+
                  " If passing single sentence, encapsulate in single element list."
            ))
        for i, sent in enumerate(text):
            if not isinstance(sent, str):
                raise(TypeError(
                    f"Input {i} not of string type."
                ))

    def naive(
          self,
          text,
          whitelist=None,
          blacklist=None
        ):
        """
        Tokenization using a Naive Approach

        Parameters
        ----------
        text : list
            A list of strings where each string is a single sentence.
        whitelist : list
            A list of characters, each an additional punctuation to take
            into account while splitting. Higher priority than blacklist.
        blacklist : list
            A list of characters, each an additional punctuation to not take
            into account while splitting. Over-ridden by whitelist if some
            character appears in both.

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens.
        """

        # Running type checking procedures
        self.__check_list_str(text)

        # Altering splitting tokens
        if blacklist is not None:
            self.__check_list_str(blacklist)
            punct = set(basic_punct)-set(blacklist)
        else:
            punct = set(basic_punct)
        if whitelist is not None:
            self.__check_list_str(whitelist)
            punct = punct.union(set(whitelist))

        # Generating tokenized text
        tokenizedText = []
        for sent in text:
            tokenizedText.append([
                word for word in re.split(
                    '(['+"".join(punct)+'\s])', sent
                ) if word not in (' ', '')
            ])
        # Explanation of regex: split on any of the characters in punct, and
        # keep the character in results. Throw away pure whitespace matches and
        # empty matches.
        return tokenizedText

    def pennTreeBank(self, text):
        """
        Tokenization using the Penn Tree Bank Tokenizer

        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """
        self.__check_list_str(text)

        tokenizer = TreebankWordTokenizer()
        tokenizedText = [ tokenizer.tokenize(sent) for sent in text ]

        return tokenizedText
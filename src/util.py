# Add your import statements here
from nltk.corpus import wordnet #for pos_tag

# Add any utility functions here

def type_check(text):
    """
    Checks if text is of type str or not
    """
    flag = 0
    flag1 = 0
    if isinstance(text,list):
        for sentences in text:
            for token in sentences:
                if isinstance(token,str):
                    flag += 1
                else:
                    flag1 = 1

    if(flag1 == 0):
        flag = 1

    if isinstance(text,str):
        flag = 1

    return flag



def pos_tag_to_wordnet(pos_token):
    """
    Converts the string pos to a wordnet part of
    speech tag to be used by the lemmatizer
    """
    if(pos_token.startswith('J')): #if tag starts with J its an adjective
        return wordnet.ADJ
    elif(pos_token.startswith('V')): #if tag starts with V its a verb
        return wordnet.VERB
    elif(pos_token.startswith('N')): #if tag starts with N its a noun
        return wordnet.NOUN
    elif(pos_token.startswith('R')): #if tag starts with R its an adverb
        return wordnet.ADV
    else:
        return wordnet.NOUN #if none of the types match just classify it as a noun (works for cases such as pronouns)

basic_punct = [
    '\'', '"', '“', '”',    # quotations
    ',', '/', ';', ':', '&', # mid-sentence punctuations
    '!', '.', '?',    # end-sentence punctuations
]

sent_end_chars = '[.?!]' #end sentence punctuations
from util import *

# Add your import statements here
import math
from collections import Counter
import pickle


class InformationRetrieval():

    def __init__(self):
        self.index = None

    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """

        index = {}
        # Fill in code here
        for idx in range(len(docs)):
            docID = docIDs[idx]
            doc = docs[idx]
            # Converting to a list of words
            terms_list = [word for sentence in doc for word in sentence]
            for term, term_frequency in list(Counter(terms_list).items()):
                if term in index.keys():
                    index[term].append([docID, term_frequency])
                else:
                    index[term] = [[docID, term_frequency]]
        self.docIDs = docIDs
        self.index = index

    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query


        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """

        doc_IDs_ordered = []
        index = self.index
        docIDs = self.docIDs
        idf_values = {}
        N = len(docIDs)
        zero_vector = {}
        # calculating the idf values for each term in vocabulary
        for term in index.keys():
            df_value = len(index[term])
            idf_values[term] = math.log10(float(N/df_value))
            zero_vector[term] = 0
        # Representing docs as Vectors with their tf-idf values
        # corresponding to each term
        doc_vectors = {}
        for docID in docIDs:
            doc_vectors[docID] = zero_vector.copy()

        # The terms which are absent in a doc are initialized to zero
        # as in the above for loop
        for term in index:
            for docID, tf in index[term]:
                doc_vectors[docID][term] = tf * idf_values[term]

        for i, query in enumerate(queries):
            print(f'Ranking for query {i+1}/{len(queries)}', end='\r')
            query_terms = [term for sentence in query for term in sentence]
            query_vector = zero_vector.copy()
            for term, tf in list(Counter(query_terms).items()):
                if term in query_vector.keys():
                    query_vector[term] = tf * idf_values[term]

            similarities = {}
            for docID in docIDs:

                try:
                    similarities[docID] = sum(doc_vectors[docID][term] * query_vector[term] for term in zero_vector) / (math.sqrt(sum(
                        doc_vectors[docID][term] ** 2 for term in zero_vector)) * math.sqrt(sum(query_vector[term] ** 2 for term in zero_vector)))
                except:
                    similarities[docID] = 0
            doc_IDs_ordered.append([docID for docID, similarity in sorted(
                similarities.items(), key=lambda item: item[1], reverse=True)])
        print("Ranking complete.")

        # Save results into a pickle file for convenience
        with open('rank_result.pkl', 'wb') as f:
            pickle.dump(doc_IDs_ordered, f)

        return doc_IDs_ordered

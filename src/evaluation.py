from util import *

from math import log2


class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The precision value as a number between 0 and 1
        """
        N_ret = len(query_doc_IDs_ordered)  # number of documents retrieved

        try:
            assert (k <= N_ret), "Number of documents retrieved less than k"

            N_ret_rel = 0  # number of relevant documents retrieved

            # [:k] as it is precision at k
            for doc_id in query_doc_IDs_ordered[:k]:

                if (int(doc_id) in true_doc_IDs):
                    N_ret_rel += 1  # increase if the retrieved doc is relevant

            precision = N_ret_rel/k  # number of relevant docs / total docs retrieved
            return precision

        except AssertionError as msg:

            print(msg)
            return -1

    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries for which the documents are ordered
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The mean precision value as a number between 0 and 1
        """
        N_queries = len(query_ids)
        precision_vals = []
        try:
            assert (len(doc_IDs_ordered) == len(query_ids)
                    ), "Mismatch in number of queries and results."

            for i in range(N_queries):  # iterating through each query

                # ranking or ordering of the ith query
                query_doc_IDs_ordered = doc_IDs_ordered[i]
                query_id = int(query_ids[i])

                # generating the true_doc_IDs using qrels
                # sample qrels entry: {"query_num": "1", "position": 2, "id": "184"}
                true_doc_IDs = []

                for qrel in qrels:  # iterating through qrels
                    if (int(qrel["query_num"]) == query_id):

                        true_doc_IDs.append(int(qrel["id"]))

                # computing precision for this query
                precision = self.queryPrecision(
                    query_doc_IDs_ordered, query_id, true_doc_IDs, k)

                precision_vals.append(precision)

            try:
                assert (len(precision_vals) !=
                        0), "Query level metric returned empty list."

                # average precision
                return sum(precision_vals)/len(precision_vals)

            except AssertionError as msg:

                print(msg)
                return -1

        except AssertionError as msg:

            print(msg)
            return -1

    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The recall value as a number between 0 and 1
        """
        N_ret = len(query_doc_IDs_ordered)  # number of documents retrieved
        N_true = len(true_doc_IDs)  # number of true documents

        try:
            assert (k <= N_ret), "Number of documents retrieved less than k"

            N_ret_rel = 0  # number of relevant documents retrieved

            for doc_id in query_doc_IDs_ordered[:k]:

                if int(doc_id) in true_doc_IDs:
                    N_ret_rel += 1

            recall = N_ret_rel/N_true  # number of relevant docs/ total true docs
            return recall

        except AssertionError as msg:

            print(msg)
            return -1

    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries for which the documents are ordered
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The mean recall value as a number between 0 and 1
        """
        N_queries = len(query_ids)
        recall_vals = []
        try:
            assert (len(doc_IDs_ordered) == len(query_ids)
                    ), "Mismatch in number of queries and results."

            for i in range(N_queries):  # iterating through each query

                # ranking or ordering of the ith query
                query_doc_IDs_ordered = doc_IDs_ordered[i]
                query_id = query_ids[i]

                # generating the true_doc_IDs using qrels
                # sample qrels entry: {"query_num": "1", "position": 2, "id": "184"}
                true_doc_IDs = []

                for qrel in qrels:  # iterating through qrels
                    if (int(qrel["query_num"]) == int(query_id)):
                        true_doc_IDs.append(int(qrel["id"]))

                # computing recall for this query
                recall = self.queryRecall(
                    query_doc_IDs_ordered, query_id, true_doc_IDs, k)
                recall_vals.append(recall)

            try:
                assert (len(recall_vals) !=
                        0), "Query level metric returned empty list."

                return sum(recall_vals)/len(recall_vals)  # average recall

            except AssertionError as msg:

                print(msg)
                return -1

        except AssertionError as msg:

            print(msg)
            return -1

    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The fscore value as a number between 0 and 1
        """
        fscore = 0
        # Note: We're not checking for any size mismatch errors as it is automatically done by the precision and recall functions

        precision = self.queryPrecision(
            query_doc_IDs_ordered, query_id, true_doc_IDs, k)  # precision
        recall = self.queryRecall(
            query_doc_IDs_ordered, query_id, true_doc_IDs, k)  # recall

        if (precision > 0 and recall > 0):  # as if they're less than 0 there's an error
            # fscore is harmonic mean of precision and recall
            fscore = 2*precision*recall/(precision+recall)

        return fscore

    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries for which the documents are ordered
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The mean fscore value as a number between 0 and 1
        """

        N_queries = len(query_ids)
        fscore_vals = []

        try:
            assert (len(doc_IDs_ordered) == len(query_ids)
                    ), "Mismatch in number of queries and results."

            for i in range(N_queries):  # iterating through each query

                # ranking or ordering of the ith query
                query_doc_IDs_ordered = doc_IDs_ordered[i]
                query_id = query_ids[i]

                # generating the true_doc_IDs using qrels
                # sample qrels entry: {"query_num": "1", "position": 2, "id": "184"}
                true_doc_IDs = []

                for qrel in qrels:  # iterating through qrels
                    if (int(qrel["query_num"]) == int(query_id)):
                        true_doc_IDs.append(int(qrel["id"]))

                # computing fscore for this query
                fscore = self.queryFscore(
                    query_doc_IDs_ordered, query_id, true_doc_IDs, k)
                fscore_vals.append(fscore)

            try:
                assert (len(fscore_vals) !=
                        0), "Query level metric returned empty list."

                return sum(fscore_vals)/len(fscore_vals)  # average fscore

            except AssertionError as msg:

                print(msg)
                return -1

        except AssertionError as msg:

            print(msg)
            return -1

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of nDCG of the Information Retrieval System
        at given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The qrels list of dictionaries 
        arg4 : int
                The k value

        Returns
        -------
        float
                The nDCG value as a number between 0 and 1
        """
        N_ret = len(query_doc_IDs_ordered)  # number of documents retrieved

        try:
            assert (k <= N_ret), "Number of documents retrieved less than k"

            # initializing
            rel_values = {}
            rel_docs = []
            dcgk = 0
            idcgk = 0

            # sample qrels entry: {"query_num": "1", "position": 2, "id": "184"}
            for qrel in true_doc_IDs:
                if (int(qrel["query_num"]) == int(query_id)):

                    id_ = int(qrel["id"])
                    # as the qrels entry is in terms of positions and not relevance (human measurement) , i.e., 1 is best instead of worst relevance
                    rel = 5-qrel["position"]
                    rel_docs.append(int(id_))
                    rel_values[int(id_)] = rel

            for i in range(1, k+1):
                doc_id = int(query_doc_IDs_ordered[i-1])

                if (doc_id in rel_docs):

                    rel = rel_values[doc_id]
                    dcgk += (2**rel-1)/log2(i+1)

            # optimal order of the relevance values
            optimal_order = sorted(rel_values.values(), reverse=True)
            N_rel = len(optimal_order)  # number of relevant documents

            for i in range(1, min(N_rel, k)+1):

                rel = optimal_order[i-1]
                # computing IDCG@k using formula given in class
                idcgk += (2**rel-1)/log2(i+1)

            try:

                assert (
                    idcgk != 0), "IDCG_k = zero. Using metric will result in division by zero."

                ndcgk = dcgk/idcgk

                return ndcgk

            except AssertionError as msg:

                print(msg)
                return -1

        except AssertionError as msg:

            print(msg)
            return -1

    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of nDCG of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries for which the documents are ordered
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The mean nDCG value as a number between 0 and 1
        """

        N_queries = len(query_ids)

        nDCG_vals = []
        try:
            assert (len(doc_IDs_ordered) == len(query_ids)
                    ), "Mismatch in number of queries and results."

            for i in range(N_queries):  # iterating through each query

                # ranking or ordering of the ith query
                query_doc_IDs_ordered = doc_IDs_ordered[i]
                query_id = int(query_ids[i])

                nDCG = self.queryNDCG(
                    query_doc_IDs_ordered, query_id, qrels, k)

                nDCG_vals.append(nDCG)
            try:
                assert (len(nDCG_vals) !=
                        0), "Query level metric returned empty list."

                return sum(nDCG_vals)/len(nDCG_vals)  # average nDCG

            except AssertionError as msg:

                print(msg)
                return -1

        except AssertionError as msg:

            print(msg)
            return -1

    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of average precision of the Information Retrieval System
        at a given value of k for a single query (the average of precision@i
        values for i such that the ith document is truly relevant)

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The average precision value as a number between 0 and 1
        """
        N_true = len(true_doc_IDs)  # number of true docs
        N_ret = len(query_doc_IDs_ordered)  # number of docs retrieved
        try:
            assert (k <= N_ret), "Number of documents retrieved less than k"

            rel_vals = []
            precision_vals = []

            for doc_id in query_doc_IDs_ordered:

                # append 1 if it is a relevant doc else append 0

                if (int(doc_id) in true_doc_IDs):
                    rel_vals.append(1)
                else:
                    rel_vals.append(0)

            for i in range(1, k+1):

                precision_i = self.queryPrecision(
                    query_doc_IDs_ordered, query_id, true_doc_IDs, i)  # computing precision @ i

                precision_vals.append(precision_i)

            precision_rel_k = []  # product of precision@k and rel@k

            for i in range(k):

                value = precision_vals[i]*rel_vals[i]
                precision_rel_k.append(value)

            try:
                assert (N_true != 0), "True document list is an empty list"

                avgP = sum(precision_rel_k)/N_true
                return avgP

            except AssertionError as msg:

                print(msg)
                return -1

        except AssertionError as msg:

            print(msg)
            return -1

    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of MAP of the Information Retrieval System
        at given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The MAP value as a number between 0 and 1
        """
        N_queries = len(query_ids)
        avg_precision_vals = []

        try:
            assert (len(doc_IDs_ordered) == len(query_ids)
                    ), "Mismatch in number of queries and results."

            for i in range(N_queries):  # iterating through each query

                # ranking or ordering of the ith query
                query_doc_IDs_ordered = doc_IDs_ordered[i]
                query_id = int(query_ids[i])

                # generating the true_doc_IDs using qrels
                # sample qrels entry: {"query_num": "1", "position": 2, "id": "184"}
                true_doc_IDs = []

                for qrel in qrels:  # iterating through qrels

                    if (int(qrel["query_num"]) == int(query_id)):
                        true_doc_IDs.append(int(qrel["id"]))

                # computing average precision for given query
                avgP = self.queryAveragePrecision(
                    query_doc_IDs_ordered, query_id, true_doc_IDs, k)
                avg_precision_vals.append(avgP)
            try:
                assert (len(avg_precision_vals) !=
                        0), "Query level metric returned empty list."

                return sum(avg_precision_vals)/len(avg_precision_vals)

            except AssertionError as msg:

                print(msg)
                return -1

        except AssertionError as msg:

            print(msg)
            return -1

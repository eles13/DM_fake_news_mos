import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

#resource: https://github.com/LouisTsiattalou/tfidf_matcher

def ngrams(string, n=3):
    # Assert string type
    assert type(string) == type("string"), "String not passed in!"
    # Remove Punctuation from the string
    string = re.sub(r"[,-./]|\sBD", r"", string)

    # Generate zip of ngrams (n defined in function argument)
    ngrams = zip(*[string[i:] for i in range(n)])
    # Return ngram list
    return ["".join(ngram) for ngram in ngrams]


class Matcher:
    
    def __init__(self, lookup_path, k_matches=5, ngram_length=3):
        """Takes two lists, returns top `k` matches from `lookup` dataset.
        This function does this by:
        - Splitting the `lookup` list into ngrams.
        - Transforming the resulting ngram list into a TF-IDF Sparse Matrix.
        - Fit a NearestNeighbours Model to the matrix using the lookup data.
        - Transform the `original` list into a TF-IDF Sparse Matrix.
        - Calculates distances to all the `n-matches` nearest neighbours
        - Then extract the `original`, `n-matches` closest lookups, and calculate
        a match score (abs(1 - Distance to Nearest Neighbour))
        :param original: List of strings to generate ngrams from.
        :type original: list (of strings), or Pandas Series.
        :param lookup: List of strings to match against.
        :type lookup: list (of strings), or Pandas Series.
        :param k_matches: Number of matches to return.
        :type k_matches: int
        :param ngram_length: Length of Ngrams returned by `tfidf_matcher.ngrams` callable
        :type ngram_length: int
        :raises AssertionError: Throws an error if the datatypes in `original` aren't strings.
        :raises AssertionError: Throws an error if the datatypes in `lookup` aren't strings.
        :raises AssertionError: Throws an error if `k_matches` isn't an integer.
        :raises AssertionError: Throws an error if k_matches > len(lookup)
        :raises AssertionError: Throws an error if ngram_length isn't an integer
        :return: Returns a Pandas dataframe with the `original` list,
            `k_matches` columns containing the closest matches from `lookup`,
            as well as a Match Score for the closest of these matches.
        :rtype: Pandas dataframe
        """
        self.k_matches = k_matches
        self.ngram_length = ngram_length
        mos_news=pd.read_csv(lookup_path)  
        self.mos_news = mos_news.drop_duplicates(subset=['id']).reset_index(drop=True)
        self.lookup=self.mos_news["title"]

        # Enforce listtype, set to lower
        self.lookup = list(self.lookup)
        self.lookup_lower = [x.lower() for x in self.lookup]

        # Set ngram length for TfidfVectorizer callable
        def ngrams_user(string, n=ngram_length):
            return ngrams(string, n)

        # Generate Sparse TFIDF matrix from Lookup corpus
        self.vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams_user)
        self.tf_idf_lookup = self.vectorizer.fit_transform(self.lookup_lower)

        # Fit KNN model to sparse TFIDF matrix generated from Lookup
        self.nbrs = NearestNeighbors(n_neighbors=k_matches, n_jobs=-1, metric="cosine").fit(
            self.tf_idf_lookup
        )

    def match(self, query):
        query = list(query)
        query = [x.lower() for x in query]
        tf_idf_original = self.vectorizer.transform(query)
        distances, lookup_indices = self.nbrs.kneighbors(tf_idf_original)
        original_name_list = []
        confidence_list = []
        index_list = []
        lookup_list = []
        # i is 0:len(original), j is list of lists of matches
        for i, lookup_index in enumerate(lookup_indices):
            original_name = query[i]
            # lookup names in lookup list
            lookups = [self.lookup[index] for index in lookup_index]
            # transform distances to confidences and store
            confidence = [1 - round(dist, 2) for dist in distances[i]]
            original_name_list.append(original_name)
            # store index
            index_list.append(lookup_index)
            confidence_list.append(confidence)
            lookup_list.append(lookups)

        # Convert to df
        df_orig_name = pd.DataFrame(original_name_list, columns=["Original Name"])

        df_lookups = pd.DataFrame(
            lookup_list, columns=["Lookup " + str(x + 1) for x in range(0, self.k_matches)]
        )
        df_confidence = pd.DataFrame(
            confidence_list,
            columns=["Lookup " + str(x + 1) + " Confidence" for x in range(0, self.k_matches)],
        )
        df_index = pd.DataFrame(
            index_list,
            columns=["Lookup " + str(x + 1) + " Index" for x in range(0, self.k_matches)],
        )

        # bind columns
        matches = pd.concat([df_orig_name, df_lookups, df_confidence, df_index], axis=1)

        # reorder columns | can be skipped
        lookup_cols = list(matches.columns.values)
        lookup_cols_reordered = [lookup_cols[0]]
        for i in range(1, self.k_matches + 1):
            lookup_cols_reordered.append(lookup_cols[i])
            lookup_cols_reordered.append(lookup_cols[i + self.k_matches])
            lookup_cols_reordered.append(lookup_cols[i + 2 * self.k_matches])
        matches = matches[lookup_cols_reordered]
        titles_matches = matches[[f'Lookup {i}' for i in range(1,6)]].values[0]
        res = self.mos_news.iloc[lookup_indices[0]].reset_index(drop=True)
        return res

if __name__ == '__main__':
    matcher = Matcher('../resources/news.csv')
    print(matcher.match(["В День России в центре Москвы изменится схема"]).title)
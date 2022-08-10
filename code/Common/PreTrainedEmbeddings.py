import numpy as np
import torch
from scipy import spatial

class PreTrainedEmbeddings:
    """PreTrained embedding vectors for words from some dictionary"""

    def __init__(self, word_to_index, word_vectors):
        """
        Args:
            word_to_index (dict): mapping from words to integers, that represent the position in word_vectors list that corresponds with the embedding vector for each word
            word_vectors (list of numpy arrays): list of embedding vectors for each word
        """
        self.word_to_index = word_to_index
        self.word_vectors = word_vectors

        # add a reverse mapping: from index in the dictionary to the actual word
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}

        # build a tree for quick lookups and nearest-neighbor queries
        self.lookup_index = spatial.KDTree(word_vectors)


    @classmethod
    def from_embeddings_file(cls, embedding_file_path):
        """
        Initializes the class from file with pre-trained vectors

        Vector file is expected to be in format:
            word0 x0_0 x0_1 ... x0_N
            word1 x1_0 x1_1 ... x1_N
            ...
            wordK xK_0 xK_1 ... xK_N

        Args:
            embedding_file_path (str): path to the file with pretrained embeddings
        Returns:
            an instance of the PreTrainedEmbeddings
        """
        # both word-to-index map and embeddings list are empty in the beginning
        word_to_index = {}
        word_vectors = []

        with open(embedding_file_path, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                # split the line
                line = line.split(" ")
                # the first value corresponds to the word
                word = line[0]
                # the rest of the values correspods to the embedding vector values
                embedding_vector = np.array([float(x) for x in line[1:]])
                # word will be added to the embeddings list on the first free location
                word_to_index[word] = len(word_vectors)
                # add the word to the embeddings list
                word_vectors.append(embedding_vector)

        return cls(word_to_index, np.stack(word_vectors))


    def get_embedding(self, word):
        """
        Returns an embedding vector for the word

        Args:
            word (str): the word that should be mapped to embedding vector
        Returns:
            an embedding (numpy.apply)
        """
        return self.word_vectors[self.word_to_index[word]]


    def get_words_closest_to_vector(self, vector, n=1):
        """
        Given a vector returns its n nearest neighbors in the constructed vector space

        Args:
            vector (numpy.ndarray): vector, which size should match the size of vectors in the self.lookup_index
            n (int): number of neighbors to return
        Returns:
            [str, str, ..., str] (list of str): words nearest in the given vector
        """
        # query function returns the distances to the nearest neighbors (array of floats) and the locations of the neighbors in the input sequence (array of integers)
        _, nn_indices = self.lookup_index.query(vector, n)

        nn_words = [self.index_to_word[neighbor_index] for neighbor_index in nn_indices]

        return nn_words


    def make_embeddings_matrix(self, word_list):
        """
        Create embedding matrix for a specified set of words

        Args:
            word_list (list): list of words in a dataset
        Returns:
            final_embeddings (numpy.ndarray): embedding matrix
        """
        # get the embedding size
        embedding_size = self.word_vectors.shape[1]

        # set up the embedding matrix
        final_embeddings = np.zeros((len(word_list), embedding_size))

        for i, word in enumerate(word_list):
            # if the word is among pretrained embeddings
            if word in self.word_to_index:
                # use the pretrained embedding vector as initial value for the embedding vector for the word
                embedding_vector = self.get_embedding(word)
            else:
                # if the word is not available among pretrained embeddings, initialize the embedding vector for the word using Xavier Uniform method
                embedding_vector = torch.ones(1, embedding_size)
                torch.nn.init.xavier_uniform_(embedding_vector)

            # set the embedding vector for the current word
            final_embeddings[i, :] = embedding_vector

        return final_embeddings

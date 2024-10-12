from chromadb import Documents, Embeddings
from chromadb.utils import embedding_functions
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform


class MyEmbeddingFunction(embedding_functions.SentenceTransformerEmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        output_embeddings = super().__call__(input)
        return output_embeddings

    def pca_transform(self, embeddings, pcac):
        if pcac > 0:
            # Fit PCA
            pca = PCA(n_components=min(pcac,len(embeddings)), svd_solver='full')
            pca.fit(embeddings)

            # Examine the explained variance
            explained_variance = pca.explained_variance_ratio_
            print(explained_variance)
            transformed_embeddings = []
            # emb = np.array(embeddings).reshape(1, -1)
            principal_component = pca.transform(embeddings)
            reconstructed_embedding = pca.inverse_transform(principal_component)
            adjusted_embedding = embeddings - reconstructed_embedding
            embedding_norm = np.linalg.norm(adjusted_embedding, axis=1, keepdims=True)
            norm_embedding = adjusted_embedding / embedding_norm
            transformed_embeddings = norm_embedding
            print("Adjusted embedding: ", adjusted_embedding)
            print("Normalized embedding: ", norm_embedding)
            print("Original similarity:\\n", 1 - pdist(embeddings, 'cosine'))
            print("Transformed similarity:\\n", 1 - pdist(transformed_embeddings, 'cosine'))
            print("Cosine similarity between original and transformed:\\n", list(1 - pdist(x, 'cosine') for x in zip(embeddings,transformed_embeddings)))
            return list(transformed_embeddings)
        return embeddings

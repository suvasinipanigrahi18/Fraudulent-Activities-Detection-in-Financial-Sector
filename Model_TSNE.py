from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def Model_tsne(features):
    # Initialize t-SNE model with desired parameters
    tsne = TSNE(n_components=2, random_state=42)

    # Fit and transform the feature vectors using t-SNE
    embedded_features = tsne.fit_transform(features)

    return embedded_features
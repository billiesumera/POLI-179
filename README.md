# Word2Vec Gender Bias Analysis

This project analyzes gender bias in word embeddings using the pre-trained Word2Vec model. It calculates the cosine similarity between profession-related words and gender-related words to measure bias. Visualizations using PCA (Principal Component Analysis) are also provided to help understand the relationships between words in a 2D space.

## Project Description

The project uses the Word2Vec model from the Gensim library, which is pre-trained on large datasets such as Google News and Wikipedia. The following professions are analyzed:
- Doctor
- Police
- Teacher
- Reporter
- Dancer

For each profession, similarity scores are calculated between the profession and gender-related words (both male and female). The results are visualized using PCA.

## Setup Instructions

### Prerequisites

- Python 3.x
- Gensim
- Matplotlib
- Scipy
- Scikit-learn

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/word2vec-gender-bias-analysis.git
    cd word2vec-gender-bias-analysis
    ```

2. Install the required libraries:
    ```bash
    pip install gensim matplotlib scipy scikit-learn
    ```

## Usage

1. Load the pre-trained Word2Vec model and libraries:
    ```python
    import matplotlib.pyplot as plt
    from

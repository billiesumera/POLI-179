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
    scipy.spatial import distance
    from sklearn.decomposition import PCA
    import gensim.downloader as api

    # Load pre-trained model
    model = api.load('word2vec-google-news-300')
    ```

2. Calculate similarity scores for each profession:
    ```python
    professions = ['doctor', 'police', 'teacher', 'reporter', 'dancer']
    females = ['female', 'woman', 'lady', 'girl', 'she', 'her']
    males = ['male', 'man', 'guy', 'boy', 'he', 'him']

    for profession in professions:
        similarities_female = []
        similarities_male = []

        for female in females:
            if profession in model.key_to_index and female in model.key_to_index:
                similarity = 1 - distance.cosine(model[profession], model[female])
                similarities_female.append(similarity)
                print(f"Similarity between '{profession}' and '{female}': {similarity:.3f}")
            else:
                similarities_female.append(0)

        for male in males:
            if profession in model.key_to_index and male in model.key_to_index:
                similarity = 1 - distance.cosine(model[profession], model[male])
                similarities_male.append(similarity)
                print(f"Similarity between '{profession}' and '{male}': {similarity:.3f}")
            else:
                similarities_male.append(0)
    ```

3. Visualize the word vectors using PCA:
    ```python
    words = professions + males + females
    word_vectors = [model[word] for word in words if word in model.key_to_index]

    # PCA for dimensionality reduction to 2D
    pca = PCA(n_components=2)
    word_vecs_2d = pca.fit_transform(word_vectors)

    # Plotting
    plt.figure(figsize=(10, 6))
    for word, coord in zip(words, word_vecs_2d):
        x, y = coord
        plt.scatter(x, y, marker='x', color='blue')
        plt.text(x + 0.1, y + 0.1, word, fontsize=9)
    plt.title('2D Visualization of Word Vectors using PCA')
    plt.show()
    ```

## Results

### Doctor

- Similarity with female-related words:
    - `doctor` and `female`: 0.133
    - `doctor` and `woman`: 0.379
    - `doctor` and `lady`: 0.260
    - `doctor` and `girl`: 0.323
    - `doctor` and `she`: 0.277
    - `doctor` and `her`: 0.234

- Similarity with male-related words:
    - `doctor` and `male`: 0.126
    - `doctor` and `man`: 0.314
    - `doctor` and `guy`: 0.227
    - `doctor` and `boy`: 0.341
    - `doctor` and `he`: 0.274

### Police

- Similarity with female-related words:
    - `police` and `female`: 0.141
    - `police` and `woman`: 0.369
    - `police` and `lady`: 0.125
    - `police` and `girl`: 0.309
    - `police` and `she`: 0.230
    - `police` and `her`: 0.228

- Similarity with male-related words:
    - `police` and `male`: 0.216
    - `police` and `man`: 0.367
    - `police` and `guy`: 0.080
    - `police` and `boy`: 0.318
    - `police` and `he`: 0.224
    - `police` and `him`: 0.253

### Teacher

- Similarity with female-related words:
    - `teacher` and `female`: 0.165
    - `teacher` and `woman`: 0.314
    - `teacher` and `lady`: 0.287
    - `teacher` and `girl`: 0.384
    - `teacher` and `she`: 0.282
    - `teacher` and `her`: 0.271

- Similarity with male-related words:
    - `teacher` and `male`: 0.145
    - `teacher` and `man`: 0.250
    - `teacher` and `guy`: 0.189
    - `teacher` and `boy`: 0.340
    - `teacher` and `he`: 0.161
    - `teacher` and `him`: 0.177

### Reporter

- Similarity with female-related words:
    - `reporter` and `female`: 0.079
    - `reporter` and `woman`: 0.239
    - `reporter` and `lady`: 0.233
    - `reporter` and `girl`: 0.199
    - `reporter` and `she`: 0.190
    - `reporter` and `her`: 0.172

- Similarity with male-related words:
    - `reporter` and `male`: 0.027
    - `reporter` and `man`: 0.193
    - `reporter` and `guy`: 0.199
    - `reporter` and `boy`: 0.183
    - `reporter` and `he`: 0.185
    - `reporter` and `him`: 0.207

### Dancer

- Similarity with female-related words:
    - `dancer` and `female`: 0.283
    - `dancer` and `woman`: 0.365
    - `dancer` and `lady`: 0.338
    - `dancer` and `girl`: 0.411
    - `dancer` and `she`: 0.293
    - `dancer` and `her`: 0.297

- Similarity with male-related words:
    - `dancer` and `male`: 0.262
    - `dancer` and `man`: 0.244
    - `dancer` and `guy`: 0.247
    - `dancer` and `boy`: 0.279
    - `dancer` and `he`: 0.146
    - `dancer` and `him`: 0.169

## Visualization
```bash
words = professions + males + females
word_vectors = [model[word] for word in words if word in model.key_to_index]
```
## PCA for dimensionality reduction to 2D
```bash
pca = PCA(n_components=2)
word_vecs_2d = pca.fit_transform(word_vectors)
```
## Plotting
```bash
plt.figure(figsize=(10, 6))
for word, coord in zip(words, word_vecs_2d):
    x, y = coord
    plt.scatter(x, y, marker='x', color='blue')
    plt.text(x + 0.1, y + 0.1, word, fontsize=9)
plt.title('2D Visualization of Word Vectors using PCA')
plt.show()
```
# Method 2 - Bing Image Downloader
- This method uses the Bing Image Downloader to collect images for each profession. The images are analyzed to see if there is gender bias in the search results.
- A sample of images is displayed for each profession.

## Setup
```bash
!pip install bing-image-downloader
!mkdir images


from bing_image_downloader import downloader
from IPython.display import Image, display
```
## Job 1 : Doctor - Image Collection

- Download images for the profession 'doctor'
```bash
downloader.download("doctor", limit=100,  output_dir='images', adult_filter_off=True, force_replace=False)
doctor_images_path = 'images/doctor'
doctor_image_files = os.listdir(doctor_images_path)
```
- Display the first 5 images as a sample
```bash
for doctor_image_file in doctor_image_files[:5]:
    display(Image(filename=os.path.join(doctor_images_path, doctor_image_file)))
from bing_image_downloader import downloader
from IPython.display import Image, display
```
## Job 2 : Police - Image Collection

- Download images for the profession 'police'
```bash
downloader.download("police", limit=100,  output_dir='images', adult_filter_off=True, force_replace=False)
police_images_path = 'images/police'
police_image_files = os.listdir(police_images_path)
supported_formats = ('.png', '.jpg', '.jpeg', '.gif')
```
- Display the first 5 images as a sample
```bash
for police_image_file in police_image_files[:5]:
    if police_image_file.lower().endswith(supported_formats):
        display(Image(filename=os.path.join(police_images_path, police_image_file)))

from bing_image_downloader import downloader
from IPython.display import Image, display
```
## Job 3 : Dancer - Image Collection

- Download images for the profession 'dancer'
```bash
downloader.download("dancer", limit=100,  output_dir='images', adult_filter_off=True, force_replace=False)
dancer_images_path = 'images/dancer'
dancer_image_files = os.listdir(dancer_images_path)
```
- Display the first 5 images as a sample
```bash
for dancer_image_file in dancer_image_files[:5]:
    display(Image(filename=os.path.join(dancer_images_path, dancer_image_file)))

from bing_image_downloader import downloader
from IPython.display import Image, display
```
## Job 4 : Teacher - Image Collection

- Download images for the profession 'teacher'
```bash
downloader.download("teacher", limit=100,  output_dir='images', adult_filter_off=True, force_replace=False)
teacher_images_path = 'images/teacher'
teacher_image_files = os.listdir(teacher_images_path)
```
- Display the first 5 images as a sample
```bash
for teacher_image_file in teacher_image_files[:5]:
    display(Image(filename=os.path.join(teacher_images_path, teacher_image_file)))

from bing_image_downloader import downloader
from IPython.display import Image, display
```
## Job 4 : Reporter - Image Collection

- Download images for the profession 'reporter'
```bash
downloader.download("reporter", limit=100,  output_dir='images', adult_filter_off=True, force_replace=False)
reporter_images_path = 'images/reporter'
reporter_image_files = os.listdir(reporter_images_path)
```
- Display the first 5 images as a sample
```bash
for reporter_image_file in reporter_image_files[:5]:
    display(Image(filename=os.path.join(reporter_images_path, reporter_image_file)))

```





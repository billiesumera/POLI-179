# Through the Lens: Exploring Gender Bias in Online Images 

Partners:    Billie Sumera (ID: A18133095)   &   Pauline Ramos (ID: A17091192)

 <img width="600" alt="percentscores" src="https://github.com/billiesumera/POLI-179/assets/166160863/853c5271-d427-4630-b9ac-a9a230a6c3c5">

Link to [Google Drive Folder]([url](https://drive.google.com/drive/folders/1zlHfmzSYK3tYoJnXmzWJl4isqDU1d7_f?usp=drive_link)) 
### Research Question
Do online images exacerbate gender bias based on profession? 
### Description of Data 

- Data 1   Top 100 image results when you search the word "doctor," “police,” “teacher,” “reporter” and “dancer” 
- Data 2   We will utilize the "word2vec-google-news-300" model, a pre-trained Word2Vec model provided by Google and accessed through the Gensim API. This model is trained on a portion of the Google News dataset, containing about 100 billion words, and includes word vectors for a vocabulary of 3 million words and phrases.

# Gender Bias Analysis

This project analyzes gender bias in word embeddings using the pre-trained Word2Vec model. It calculates the cosine similarity between profession-related words and gender-related words to measure bias. Visualizations using PCA (Principal Component Analysis) are also provided to help understand the relationships between words in a 2D space. We will compare these results to gender distribution accross searched images.

## Project Description

### Method 1 Overview
Method 1 of this project uses the Word2Vec model from the Gensim library, which is pre-trained on large datasets such as Google News and Wikipedia. The following professions are analyzed:
- Doctor
- Police
- Teacher
- Reporter
- Dancer

For each profession, similarity scores are calculated between the profession and gender-related words (both male and female). The results are visualized using PCA.

Next, for comparison with Method 2, percentages were calculated by taking these similarity scores and normalizing them to a 0-100 scale, where 0 represents maximum similarity to female words and 100 represents maximum similarity to male words.

### Method 2 Overview
Method 2 of this project uses the Bing Image Downloader to collect images for each profession. Each image is fed through Fairface gender detection model. The gender distribution is calculated into a percentage where 100 means all male photos.

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
```python
words = professions + males + females
word_vectors = [model[word] for word in words if word in model.key_to_index]
```
## PCA for dimensionality reduction to 2D
```python
pca = PCA(n_components=2)
word_vecs_2d = pca.fit_transform(word_vectors)
```
## Plotting
```python
plt.figure(figsize=(10, 6))
for word, coord in zip(words, word_vecs_2d):
    x, y = coord
    plt.scatter(x, y, marker='x', color='blue')
    plt.text(x + 0.1, y + 0.1, word, fontsize=9)
plt.title('2D Visualization of Word Vectors using PCA')
plt.show()
```

## Method 2: FairFace Gender Prediction
**Step 1: Bing Image Collection and Libraries**
- This method uses the Bing Image Downloader to collect images for each profession. The images are analyzed to see if there is gender bias in the search results.
- A sample of images is displayed for each profession.
- This provides a simple way to download and display images of 'doctor', 'police', 'dancer', 'teacher', and 'reporter' using the Bing Image Downloader.

**Step 2: Using the FairFace Model**

The model can be described by the following steps:

1. **Preprocess the Image:** Resize and normalize the input image to 224x224 pixels.
2. **Forward Pass:** Pass the image through the pre-trained ResNet34 model.
- The FairFace model uses a pre-trained ResNet34 architecture, which is a type of Convolutional Neural Network (CNN).
    -**Dlib Face Detection Model** - mmod_human_face_detector.dat
        - Purpose: Detects faces in an image
    - **Dlib Shape Predictor Model** - shape_predictor_5_face_landmarks.dat
        - Purpose: Predicts key landmarks on the detected faces for alignment
3. **Extract Predictions:** Obtain scores for race, gender, and age categories.
4. **Softmax:** Convert scores to probabilities.
    - ensures that the output values are between 0 and 1 and sum to 100%.
5. **Determine Predicted Class:** The predicted class for each attribute (race, gender, age) is the one with the highest probability.
6. **Map to Labels:** Convert class indices to meaningful labels.

Install relevant libraries and packages
```python
!pip install pandas
!pip install pillow

import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import shutil
```
Download Bing-Image Downloader
```python
! pip install bing-image-downloader
from bing_image_downloader import downloader
```
Mount the Google Drive
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive/')

! mkdir images
``` 
## FairFace Model and Libraries
Clone the Fairface Repository from GitHub
```python
! git clone https://github.com/dchen236/FairFace.git
! pip install dlib
%cd FairFace

!ls
```
Mount Google Drive
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)
```

Define the paths
```python
# Define the paths
import os
import shutil
```

Define the path to the Google Drive project folder
```python
# Define the path to the Google Drive project folder
drive_project_path = '/content/drive/My Drive/POLI_179_Project'
```

Define the model file paths in Google Drive
```python
# Define the model file paths in Google Drive
model_files = [
    'fairface_alldata_20191111.pt',
    'fairface_alldata_4race_20191111.pt'
]
```
Define the destination directory in FairFace project
```python
# Define the destination directory in FairFace project
fairface_model_dir = '/content/FairFace/fair_face_models'
```

Create the destination directory if it doesn't exist
```python
# Create the destination directory if it doesn't exist
os.makedirs(fairface_model_dir, exist_ok=True)
```

Copy each model file from Google Drive to the FairFace directory
```python
# Copy each model file from Google Drive to the FairFace directory
for model_file in model_files:
    model_file_drive_path = os.path.join(drive_project_path, model_file)
    model_file_fairface_path = os.path.join(fairface_model_dir, model_file)

    shutil.copy(model_file_drive_path, model_file_fairface_path)
    print(f"Model file {model_file} copied to: {model_file_fairface_path}")
```
Define the path to the test outputs CSV file
```python
# Define the path to the test outputs CSV file
test_outputs_csv_path = '/content/FairFace/test_outputs.csv'

# Clear the contents of the file
with open(test_outputs_csv_path, 'w') as file:
    file.truncate(0)

print(f"Cleared the contents of {test_outputs_csv_path}")
```
Clear the detected_faces directory
```python
# Clear the detected_faces directory
if os.path.exists(detected_faces_dir):
    for file_name in os.listdir(detected_faces_dir):
        file_path = os.path.join(detected_faces_dir, file_name)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    print(f"Cleared the directory: {detected_faces_dir}")
else:
    os.makedirs(detected_faces_dir)
    print(f"Created the directory: {detected_faces_dir}")
```
## Job 1: Doctor
Download Top 100 photos for 'doctor'
```python
from bing_image_downloader import downloader
downloader.download("doctor", limit=100,  output_dir='images', adult_filter_off=True, force_replace=False)
```
```python
!ls 'images'/'doctor'
```
Mount Google Drive
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

# Define the correct directory containing the images
doctor_image_dir = 'images/doctor'

# Verify the directory exists
if os.path.exists(doctor_image_dir):
    # Get the list of image file names
    doctor_image_files = os.listdir(doctor_image_dir)

    # Create a DataFrame with the full image paths
    doctor_df = pd.DataFrame([{'img_path': os.path.join(doctor_image_dir, img)} for img in doctor_image_files])

    # Define the path to save the CSV file in Google Drive
    doctor_csv_path = '/content/drive/My Drive/doctor_images.csv'

    # Save the DataFrame as a CSV file
    doctor_df.to_csv(doctor_csv_path, index=False)

    print(f"DataFrame saved as CSV file at: {doctor_csv_path}")
else:
    print(f"The directory {doctor_image_dir} does not exist.")
```
## Gender Detection: Doctor
Using FairFace Model to classify the image as 'female' or 'male'
```python
# Define the path to the CSV file in Google Drive
doctor_csv_path = '/content/drive/My Drive/doctor_images.csv'

# Define the path to the FairFace directory
fairface_dir = '/content/FairFace/'

# Verify the FairFace directory exists
if os.path.exists(fairface_dir):
    # Define the destination path for the CSV file in the FairFace directory
    fairface_csv_path = os.path.join(fairface_dir, 'doctor_images.csv')

    # Copy the CSV file to the FairFace directory
    shutil.copy(doctor_csv_path, fairface_csv_path)

    print(f"CSV file copied to: {fairface_csv_path}")
else:
    print(f"The directory {fairface_dir} does not exist.")
```
Defining the path for the CSV file of saved photos
```python
# Define the path to the test outputs CSV file
test_outputs_csv_path = '/content/FairFace/test_outputs.csv'

# Clear the contents of the file
with open(test_outputs_csv_path, 'w') as file:
    file.truncate(0)

print(f"Cleared the contents of {test_outputs_csv_path}")
```
Using the model to predict the images' gender
```python
%cd /content/FairFace

!python3 predict.py --csv "doctor_images.csv"
```
Filtering the images identified with detected faces
```python
# Define the path to the test outputs CSV file
test_outputs_csv_path = '/content/FairFace/test_outputs.csv'

# Read the CSV file into a DataFrame
predictions_df = pd.read_csv(test_outputs_csv_path)

# Function to display an image with a title
def display_image_with_title(image_path, title):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')  # Hide the axis
    plt.show()

# Define the path to the directory where detected faces are saved
detected_faces_dir = '/content/FairFace/detected_faces'

# Use the correct column names from the CSV
image_path_column = 'face_name_align'  # Correct column name for image paths
gender_column = 'gender'  # Column name for gender predictions

# Filter and display only the detected faces from "doctor" images with gender predictions
doctor_image_prefix = "Image_"
for index, row in predictions_df.iterrows():
    # Construct the file name and path
    image_name = os.path.basename(row[image_path_column])
    if image_name.startswith(doctor_image_prefix):
        image_path = os.path.join(detected_faces_dir, image_name)
        gender_prediction = row[gender_column]
        title = f"Gender: {gender_prediction}"
        print(f"Displaying image: {image_name} with gender prediction: {gender_prediction}")
        display_image_with_title(image_path, title)
```
Displaying the gender distribution from the model
```python
# Define the path to the test outputs CSV file
test_outputs_csv_path = '/content/FairFace/test_outputs.csv'

# Read the CSV file into a DataFrame
predictions_df = pd.read_csv(test_outputs_csv_path)

# Display the first few rows to verify
print(predictions_df.head())

# Calculate the gender distribution
gender_distribution = predictions_df['gender'].value_counts()

# Display the gender distribution
print(gender_distribution)

# Plot the gender distribution
plt.figure(figsize=(8, 6))
gender_distribution.plot(kind='bar', color=['blue', 'pink'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
```
# Results

<img width="801" alt="percentscores" src="https://github.com/billiesumera/POLI-179/assets/166160863/fe32e4ec-3f09-4bdf-a5d4-14037893f572">



 @inproceedings{karkkainenfairface,
  title={FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation},
  author={Karkkainen, Kimmo and Joo, Jungseock},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2021},
  pages={1548--1558}
}
## References
Guilbeault, D., Delecourt, S., Hull, T. et al. Online images amplify gender bias. Nature 626, 1049–1055 (2024). https://doi.org/10.1038/s41586-024-07068-x

https://huggingface.co/Dhrumit1314/Age_and_Gender_Detection/blob/main/Facial_Age_and_Gender_Detection.py


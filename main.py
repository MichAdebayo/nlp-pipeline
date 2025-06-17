import kagglehub
import pandas as pd
from nltk_pipeline import nltk_pipeline

# Download latest version
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

# Load the dataset into a pandas DataFrame
df = pd.read_csv(f"{path}/IMDB Dataset.csv")    

# Apply the pipeline to the 'review' column
df['processed_review'] = df['review'].apply(nltk_pipeline)

# View the first processed review in the DataFrame
print("Initial first review:", df['review'][0])
print("Processed first review:", df['processed_review'][0])

# Save the processed DataFrame to a new CSV file
df.to_csv("processed_reviews_2.csv", index=False)
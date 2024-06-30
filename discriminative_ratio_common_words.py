import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from collections import Counter
import pandas as pd
from collections import Counter
import os
from loguru import logger

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    text = text.replace("NEW_PARAGRAPH", "")
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    #words = [word for word in words if word not in stop_words]
    tagged_words = pos_tag(words)
    # Filter for content words and remove stop words
    content_words = [word for word, tag in tagged_words if tag.startswith(('N', 'V', 'J', 'R')) and word not in stop_words]
    return content_words
    

def save_ratio_as_csv(df_discriminative_ratio):
    file_name = f"discriminative_ratio_{target}_{comparison}.csv"
    current_directory = os.path.dirname(__file__)
    file_path = os.path.join(current_directory, file_name)
    df_discriminative_ratio.to_csv(file_path, index=False)
    logger.info(f"DataFrame saved to: {file_path}")
    

# Load the dataset
df = pd.read_csv('allsides-df.csv')

# Filter rows where content is not null or empty string
df = df[df['content'].notna() & (df['content'] != '')]

# Specify the columns you want to keep
columns_to_keep = ['title', 'story_group', 'time', 'author', 'bias', 'source', 'url', 'topic', 'content']

# Drop the rest of the columns, keeping only the specified ones
filtered_df = df[columns_to_keep]

# Filter
final_df = filtered_df[filtered_df['bias'].isin(['Left', 'Right', 'Center'])]

# 'target' represents the frequency in the target text type (numerator), and
# 'comparison' represents the frequency in the comparison text type (denominator)

target = 'Right'
comparison = 'Left'

# Sample data preparation
texts_target = final_df[final_df['bias'] == target]['content'].tolist()
texts_comparison = final_df[final_df['bias'] == comparison]['content'].tolist()

# Preprocess the texts
logger.info("Starting preprocessing for text.")
target_texts = [preprocess(text) for text in texts_target]
comparison_texts = [preprocess(text) for text in texts_comparison]
logger.info("Finished preprocessing for text.")

# Flatten the lists of words
words_target = [word for text in target_texts for word in text]
words_comparison = [word for text in comparison_texts for word in text]

# Calculate word frequencies
freq_target = Counter(words_target)
freq_comparison = Counter(words_comparison)

# Calculate total number of words
total_words_target = sum(freq_target.values())
total_words_comparison = sum(freq_comparison.values())

# Find common words that appear in both lists
common_words = set(freq_target.keys()).intersection(set(freq_comparison.keys()))

# Calculate normalized frequencies for common words
normalized_freq_target = {word: freq_target[word] / total_words_target for word in common_words}
normalized_freq_comparison = {word: freq_comparison[word] / total_words_comparison for word in common_words}

# Calculate the discriminativeness ratio
discriminativeness_ratio = {
    word: (normalized_freq_target[word] / normalized_freq_comparison[word]) if normalized_freq_comparison[word] != 0 else float('inf')
    for word in common_words
}

# Sort the discriminativeness ratios from highest to lowest
sorted_discriminativeness_ratio = sorted(discriminativeness_ratio.items(), key=lambda item: item[1], reverse=True)

df_discriminative_ratio = pd.DataFrame(sorted_discriminativeness_ratio, columns=['word', 'ratio'])

save_ratio_as_csv(df_discriminative_ratio)
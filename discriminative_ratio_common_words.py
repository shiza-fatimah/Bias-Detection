import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from collections import Counter
import pandas as pd
import os
from loguru import logger
import contractions

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

lemmatizer = WordNetLemmatizer()

def preprocess(text,freq_threshold, word_freq):
    
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]
     # Filter out words that are less frequent than the threshold
    words = [word for word in words if word_freq[word] >= freq_threshold]
    # Remove single letters and non-alphabetic words
    words = [word for word in words if len(word) > 1 and word.isalpha()]
    
    return words
        

# Load the dataset
df = pd.read_csv('allsides-df-processed.csv')
df = df.dropna(subset=['content'])
df = df.dropna(subset=['bias'])

# Specify the columns you want to keep
columns_to_keep = ['bias', 'content']

# Drop the rest of the columns, keeping only the specified ones
filtered_df = df[columns_to_keep]

# Combine all texts to calculate word frequencies
all_texts = filtered_df['content'].tolist()
all_words = [word for text in all_texts for word in preprocess(text, 0, Counter())] 
word_freq = Counter(all_words)

# Set the frequency threshold
freq_threshold = 10

# Filter
filtered_df = filtered_df[filtered_df['bias'].isin(['Left', 'Right', 'Center'])]

# 'target' represents the frequency in the target text type (numerator), and
# 'comparison' represents the frequency in the comparison text type (denominator)

target = 'Left'
comparison = 'Right'

# Sample data preparation
texts_target = filtered_df[filtered_df['bias'] == target]['content'].tolist()
texts_comparison = filtered_df[filtered_df['bias'] == comparison]['content'].tolist()


# Preprocess the texts
logger.info("Starting preprocessing for text.")
texts_target = [preprocess(text, freq_threshold, word_freq) for text in texts_target]
texts_comparison = [preprocess(text, freq_threshold, word_freq) for text in texts_comparison]
logger.info("Finished preprocessing for text.")


# Flatten the lists of words
words_target = [word for text in texts_target for word in text]
words_comparison = [word for text in texts_comparison for word in text]

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

# Print the first 5 results
print("Top 5 Discriminativeness Ratios of common words:")
for word, ratio in sorted_discriminativeness_ratio[:5]:
    print(f"Word: {word}, Discriminativeness Ratio: {ratio:.4f}")

# Print the last 5 results
print("\nBottom 5 Discriminativeness Ratios of common words:")
for word, ratio in sorted_discriminativeness_ratio[-5:]:
    print(f"Word: {word}, Discriminativeness Ratio: {ratio:.4f}")
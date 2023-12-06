import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/Reddit_Data.csv')


data.rename(columns={'clean_comment': 'text', 'category': 'sentiment'}, inplace=True)

# Data preprocessing
data.dropna(subset=['text'], inplace=True)  # Drop rows with missing text data

X = data['text']
y = data['sentiment']

# Label encoding for sentiment classes
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize and pad sequences
max_words = 5000
max_sequence_length = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Create word cloud
all_text_combined = ' '.join(X_train)
wordcloud_combined = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(all_text_combined)

# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_combined, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud for All Sentiments")
plt.show()

# Create a pie chart for sentiment distribution
sentiment_counts = np.bincount(y_train)
sentiment_labels = label_encoder.classes_
plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts, labels=sentiment_labels, autopct='%1.1f%%', startangle=140)
plt.title("Sentiment Distribution")
plt.show()

# Create a bar graph for sentiment distribution
sns.countplot(x=y_train, palette="Set2")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Distribution")
plt.xticks(np.arange(num_classes), sentiment_labels, rotation=45)
plt.show()

word_counts = tokenizer.word_counts
word_counts_sorted = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))
top_words = list(word_counts_sorted.keys())[:20]
top_word_counts = list(word_counts_sorted.values())[:20]

plt.figure(figsize=(10, 6))
sns.barplot(x=top_word_counts, y=top_words, palette="viridis")
plt.xlabel("Word Count")
plt.ylabel("Words")
plt.title("Top 20 Word Frequencies")
plt.show()

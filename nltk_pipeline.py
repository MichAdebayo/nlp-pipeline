# -*- coding: utf-8 -*-
import string
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import ssl
try:
    # Create an unverified HTTPS context if necessary
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Function to process text data through a series of transformations
def nltk_pipeline(text: str) -> str:
    """Process text data through a series of transformations.

    Args:
        text (str): The input text to process.

    Returns:
        str: The processed text.
    """

    def text_to_lowercase(text: str) -> str :
        return text.lower()

    def convert_number(text) -> str:
        p = inflect.engine()
        temp_str = text.split()
        new_str = []

        for word in temp_str:
            if word.isdigit():
                new_str.append(p.number_to_words(word)) 
            else:
                new_str.append(word)

        temp_str = ' '.join(new_str)
        return temp_str

    def remove_punctuation(text: str) -> str:
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_whitespace(text: str) -> str:
        return " ".join(text.split())

    def remove_stopwords(text: str) -> str:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        return " ".join([word for word in word_tokens if word not in stop_words])

    def stem_text(text: str) -> str:
        stemmer = PorterStemmer()
        word_tokens = word_tokenize(text)
        return " ".join([stemmer.stem(word) for word in word_tokens])

    def lemmatize_text(text: str) -> list[str]:
        lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(text)
        return [lemmatizer.lemmatize(word) for word in word_tokens]

    text = text_to_lowercase(text)
    text = convert_number(text)
    text = remove_punctuation(text)
    text = remove_whitespace(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    text = lemmatize_text(text) # type: ignore

    return text
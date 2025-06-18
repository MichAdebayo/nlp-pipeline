import spacy
import string
import inflect
import re
from typing import Match


# Load the built-in English model
nlp = spacy.load("en_core_web_sm")


def spacy_pipeline(text: str) -> list[str]:
    """Process text data through a series of transformations using SpaCy.
    
    This function replicates the NLTK pipeline behavior using SpaCy components.

    Args:
        text (str): The input text to process.

    Returns:
        list[str]: The processed text as a list of tokens (lemmatized words).
    """
    
    def text_to_lowercase(text: str) -> str:
        return text.lower()
    
    def convert_number(text: str) -> str:
        p = inflect.engine()
        temp_str = text.split()
        new_str = []
        
        for word in temp_str:
            if word.isdigit():
                # Convert string to int for inflect library
                try:
                    # Type ignore for inflect library typing issue
                    word_as_number = p.number_to_words(word)  # type: ignore
                    new_str.append(word_as_number)
                except (ValueError, TypeError):
                    new_str.append(word)
            else:
                new_str.append(word)
        
        return ' '.join(new_str)
    
    def remove_punctuation(text: str) -> str:
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_whitespace(text: str) -> str:
        return " ".join(text.split())
    
    # Convert to lowercase
    text = text_to_lowercase(text)
    
    # Convert numbers to words
    text = convert_number(text)
    
    # Remove punctuation
    text = remove_punctuation(text)
    
    # Remove extra whitespace
    text = remove_whitespace(text)

    # Use SpaCy for tokenization, stopword removal, and lemmatization
    doc = nlp(text)
    
    # Filter tokens: remove stopwords, punctuation, spaces, and apply lemmatization
    processed_tokens = []
    for token in doc:
        # Skip stopwords, punctuation, and whitespace
        if not token.is_stop and not token.is_punct and not token.is_space and token.text.strip():
            # Apply stemming-like effect by using lemma and making it lowercase
            lemma = token.lemma_.lower()
            # Simple stemming approximation (SpaCy doesn't have built-in stemming)
            # This mimics the Porter Stemmer behavior to some degree
            stemmed_lemma = simple_stem(lemma)
            processed_tokens.append(stemmed_lemma)
    
    return processed_tokens

def simple_stem(word: str) -> str:
    """Simple stemming function to approximate Porter Stemmer behavior.
    
    This is a simplified stemming approach since SpaCy doesn't include
    stemming by default. For production use, you might want to integrate
    a proper stemming library.
    
    Args:
        word (str): The word to stem.
        
    Returns:
        str: The stemmed word.
    """
    # Basic suffix removal rules (simplified Porter Stemmer approach)
    suffixes = [
        'ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment', 
        'able', 'ible', 'ant', 'ent', 'ive', 'ful', 'less', 'ous'
    ]
    
    word = word.lower()
    
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    
    return word

def spacy_pipeline_advanced(text: str) -> list[str]:
    """Advanced SpaCy pipeline with more sophisticated processing.
    
    This version leverages more SpaCy features like POS tagging and
    named entity recognition for better text processing.

    Args:
        text (str): The input text to process.

    Returns:
        list[str]: The processed text as a list of tokens.
    """
    
    def text_to_lowercase(text: str) -> str:
        return text.lower()
    
    def convert_number(text: str) -> str:
        p = inflect.engine()
        # Use regex to find numbers more accurately
        def replace_number(match: Match[str]) -> str:
            number_str = match.group()
            return p.number_to_words(number_str)  # type: ignore
        
        return re.sub(r'\b\d+\b', replace_number, text)
    
    # Convert to lowercase
    text = text_to_lowercase(text)
    
    # Convert numbers to words
    text = convert_number(text)
    
    # Process with SpaCy
    doc = nlp(text)
    
    processed_tokens = []
    for token in doc:
        # More sophisticated filtering
        if (not token.is_stop and 
            not token.is_punct and 
            not token.is_space and 
            not token.like_url and 
            not token.like_email and
            token.is_alpha and  # Only alphabetic tokens
            len(token.text) > 1):  # Skip single characters
            
            # Normalize text with lemma
            lemma = token.lemma_.lower()
            processed_tokens.append(lemma)
    
    return processed_tokens


# Example usage and comparison
if __name__ == "__main__":
    sample_text = "This movie was absolutely amazing! I'd rate it 10/10. The cinematography was breathtaking."
    
    print("Original text:")
    print(sample_text)
    print("\nSpaCy Pipeline (Basic):")
    result_basic = spacy_pipeline(sample_text)
    print(result_basic)
    print("\nSpaCy Pipeline (Advanced):")
    result_advanced = spacy_pipeline_advanced(sample_text)
    print(result_advanced)

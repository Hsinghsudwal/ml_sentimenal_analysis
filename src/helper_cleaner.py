# NLP Libraries
import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from langdetect import detect, LangDetectException

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)


def data_cleaning( text,
        keep_hashtags=True,
        normalize_repeats=True,
        max_repeat=2):
        
        EMOTICON_MAP = {
            r":\)": " smile ",
            r":-\)": " smile ",
            r":D": " laugh ",
            r";\)": " wink ",
            r":\(": " sad ",
            r":-\(": " sad ",
            r":'\(": " cry "
        }

        # raw_text = text
        if not isinstance(text, str):
            return ""

        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'@\w+', ' user ', text)
        if keep_hashtags:
            text = re.sub(
                r'#(\w+)',
                lambda m: ' '.join(re.findall(r'[A-Z]?[a-z]+', m.group(1))) or m.group(1),
                text
            )
        else:
            text = re.sub(r'#\w+', ' ', text)

        for pattern, replacement in EMOTICON_MAP.items():
            text = re.sub(pattern, replacement, text)

        text = emoji.demojize(text, delimiters=(" ", " "))

        text = re.sub(r'\d+(\.\d+)?', ' number ', text)

        if normalize_repeats:
            text = re.sub(r'(.)\1{2,}', r'\1' * max_repeat, text)

        text = text.lower()
        text = re.sub(r'[-–—]', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

        # return ' '.join(tokens)
        clean_text = ' '.join(tokens)
        return clean_text
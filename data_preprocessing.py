import pandas as pd
import re
from loguru import logger
from nltk.corpus import stopwords

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def preprocess_source_column(self):
        """
        From the source column which contains name of news portal
        - remove the words inside () at the end
        - remove the word 'news'
        - remove the word 'digital'
        
        """
        logger.info("Preprocessing source column")
        
        self.df['source'] = self.df['source'].str.replace(r'\(.*?\)', '', regex=True)
        self.df['source'] = self.df['source'].str.replace('news', '', regex=False)
        self.df['source'] = self.df['source'].str.replace('digital', '', regex=False)
        self.df['source'] = self.df['source'].str.strip()
        
        logger.info("Preprocessing source column done")
        
    def remove_specific_rows(self):
        """
        Delete the row where link contains 'huffingtonpost' and text contains the word 'paywall' or 'paywalls'
        Delete the row where link contains bloomberg.com and text starts with "Why did this happen?"
        Because in these two cases the whole text is just advertisement and no news is included.
        
        """
        logger.info("Removing rows from huffingtonpost and bloomberg")
        
        self.df = self.df[~((self.df['url'].str.contains('huffingtonpost')) & 
                            (self.df['content'].str.contains(r'\bpaywall\b|\bpaywalls\b')))]
        self.df = self.df[~((self.df['url'].str.contains('bloomberg')) & 
                            (self.df['content'].str.startswith('why did this happen?')))]
        
        logger.info("Done Removing rows from huffingtonpost and bloomberg")
        
    def _remove_vox_paragraphs(self, text: str) -> str:
        """
        Remove paragraphs from Vox articles that start with 'Will you' and everything after
        that paragraph, because it only contains subscription messages.
        
        """
        
        paragraphs = text.split('new_paragraph')
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip().startswith("will you"):
                return 'new_paragraph'.join(paragraphs[:i])
        return text

    def _remove_motherjones_paragraphs(self, text: str) -> str:
        """
        Remove paragraphs from Vox articles that start with 'subscribe to the mother jones daily'
        and everything after that paragraph, because it only contains subscription messages.
        
        """
        paragraphs = text.split('new_paragraph')
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip().startswith("subscribe to the mother jones daily"):
                return 'new_paragraph'.join(paragraphs[:i])
        return text

    def _remove_ad_paragraphs(self, text: str) -> str:
        """
        Remove paragraphs containing certain advertisement-related words.
        
        """
        ad_patterns = [
            'paywall', 'paywalls', 'subscribe', 'subscription', 'login', 'log in', 'log-in', 
            'sign up', 'sign-up', 'signup', 'click', 'clicking', 'terms of service', 'privacy policy', 
            'download', 'unsubscribe', 'editor’s note:', 'journalism free'
        ]
        pattern = '|'.join(ad_patterns)
        paragraphs = text.split('new_paragraph')
        paragraphs = [p for p in paragraphs if not re.search(pattern, p, re.IGNORECASE)]
        return 'new_paragraph'.join(paragraphs)

    def _remove_unwanted_words(self, text: str) -> str:
        """
        Remove specific unwanted words and tokens from the text including news portal names.
        
        """
        unwanted_words = [
            'LOADING', 'ERROR', 'Advertisement', 'new_paragraph', 'Related stories', 'Further reading'
        ]
        news_sources = self.df['source'].unique()
        for source in news_sources:
            unwanted_words.append(source.lower())
            
        for word in unwanted_words:
            text = text.replace(word.lower(), '')
              
        return text
    
    def remove_punctuation_stopwords_digits(self, text: str) -> str:
        """
        Remove punctuation, stopwords, digits, etc.
        
        """
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
        text = re.sub(r'\[.*?\]', '', text) # Remove text inside a square bracket
        text = re.sub(r'[^a-zA-Z0-9\s-]', '', text) # Remove non-English characters from text,
        stop_words = set(stopwords.words('english'))
        words = [word for word in text.split() if word.lower() not in stop_words and not word.isdigit()]
        return ' '.join(words).strip()
    
    def _preprocess_content(self, row) -> str:
        """
        Apply preprocessing steps to the content column.
        
        """
        text = row['content']

        # Remove unwanted paragraphs for Vox articles
        logger.info("Removing vox paragraphs")
        if 'vox' in row['url']:
            text = self._remove_vox_paragraphs(text)

        # Remove unwanted paragraphs for Mother Jones articles
        logger.info("Removing Mother Jones paragraphs")
        if 'motherjones' in row['url']:
            text = self._remove_motherjones_paragraphs(text)

        # Remove advertisement-related paragraphs
        logger.info("Removing ad paragraphs")
        text = self._remove_ad_paragraphs(text)

        # Remove specific unwanted words and tokens
        logger.info("Removing unwanted words")
        text = self._remove_unwanted_words(text)
        
        #Remove punctuation, stopwords, digits, etc.
        logger.info("Removing punctuation, stopwords, digits, etc.")
        #text = self.remove_punctuation_stopwords_digits(text)

        return text

    def preprocess(self) -> pd.DataFrame:
        """Run all preprocessing steps on the DataFrame."""
        logger.info("Starting preprocessing.")

        # Convert all text to lowercase
        logger.info("Converting all text to lowercase")
        self.df['content'] = self.df['content'].str.lower()
        self.df['source'] = self.df['source'].str.lower()
        
        # Remove rows with containing NaN values or only whitespace and empty rows whitespace
        logger.info("Removing nan and empty rows")
        #self.df = self.df[self.df['content'].notna() & self.df['content'].str.strip().astype(bool)]
        #self.df = self.df[self.df['bias'].notna() & self.df['bias'].str.strip().astype(bool)]
        self.df = self.df.dropna(subset=['content'])
        self.df = self.df.dropna(subset=['bias'])
        
        # Remove spaces before and after 'new_paragraph'
        logger.info("Removing spaces before and after 'new_paragraph")
        self.df['content'] = self.df['content'].str.replace(r'\s*new_paragraph\s*', 'new_paragraph', regex=True)

        # Preprocess the 'source' column
        self.preprocess_source_column()

        # Remove specific rows based on conditions
        self.remove_specific_rows()

        # Preprocess content based on URL
        self.df['content'] = self.df.apply(self._preprocess_content, axis=1)

        logger.info("Preprocessing completed successfully.")
        return self.df


if __name__ == "__main__":
    df = pd.read_csv('allsides-df.csv')
    preprocessor = Preprocessor(df)
    processed_df = preprocessor.preprocess()
    processed_df.to_csv('allsides-df-processed.csv', index=False)
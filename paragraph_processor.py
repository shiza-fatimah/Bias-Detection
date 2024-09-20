import pandas as pd

class ParagraphProcessor:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    def get_first_quarter(self):
        """Return the first 1/4 of the text from the 'content' column."""
        return self.dataframe['content'].apply(lambda content: content[:len(content) // 4])
    
    def get_first_half(self):
        """Return the first 1/2 of the text from the 'content' column."""
        return self.dataframe['content'].apply(lambda content: content[:len(content) // 2])

    def get_first_three_quarters(self):
        """Return the first 3/4 of the text from the 'content' column."""
        return self.dataframe['content'].apply(lambda content: content[:(3 * len(content)) // 4])

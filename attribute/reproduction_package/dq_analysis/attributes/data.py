import pandas as pd
from ast import literal_eval

path = ''


class Data:
    """
    Load and manipulate a dataset

    Example:
    df = Data('Big-Vul').get_dataset()
    """
    def __init__(self, dataset: str):
        """
        Load a dataset.
        """
        # Supported options
        # 增加支持的数据集
        if dataset not in ['BigVul', 'D2A', 'Devign', 'Juliet', 
                           'toy', 'Gen-C', 'benchmark', 'Detect0day', 
                           'VDSimilar', 'VulPecker', 'Pan-et-al', 
                           'Ponta-et-al', 'Reis-et-al', 'PatchDB',
                           'Le', 'CVEFixes', 'ReVeal', 
                           'VulCNN', 'SPI', 'Magma', 'SECBENCH', 
                           'diversevul', 'CrossVul', 'Funded', 
                           'VulDeeLocator', 'SySeVR', 'VulDeePecker', 
                           'BigVul+Devign','BigVul+CVEFixes', 
                           'Linetal.', 'REEF-data', 'TransferRepresentationLearning',
                           'muVulDeePecker', 'VulnPatchPairs',
                           'ReposVul']:
            return "Error: non-supported dataset"

        # Load the data
        self.data_name = dataset
        self.df = pd.read_csv(f'{path}{dataset}/dataset.csv')

    def get_dataset(self):
        """Return loaded dataset."""
        return self.df.fillna('')

    def get_metadata(self):
        """Return metadata."""
        return pd.read_csv(f'{path}{self.data_name}/metadata.csv')

    def get_features(self):
        """
        Get encoded nlp features for the loaded dataset.
        Returns a dataframe containing the features.
        DF columns: ID, Features, Vulnerable
        """
        df = pd.read_parquet(f'{path}{self.data_name}/features_cb.parquet')
        df['Features'] = df['Features'].apply(
            lambda x: [float(y) for y in x.strip("[]").split(', ')])
        return df

    def get_tokens(self):
        """
        Get lexicographically parsed tokens of a dataset.
        Returns a dataframe containing the features.
        Tokens are stored per entry as a list of tuples:
            (token.value, token.name)
        DF columns: ID, Token, Vulnerable
        """
        return pd.read_csv(
            f'{path}{self.data_name}/tokens.csv',
            converters={'Token': literal_eval})


if __name__ == '__main__':
    # TEST
    data = Data(dataset='Juliet')
    df = data.get_dataset()
    print(df)

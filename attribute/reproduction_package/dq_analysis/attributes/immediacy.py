"""
Analyse the immediacy of datasets.
"""
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from sctokenizer import CTokenizer
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import jensenshannon, cosine
from data import Data
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# from dq_analysis.svp.tsne import plot_embeddings

if len(sys.argv) < 2:
    print("Usage: python dq_analysis/attributes/immediacy.py [prepare/measure] <dataset>")
    exit()


def tokenize(text):
    tokenizer = CTokenizer()
    try:
        tokens = tokenizer.tokenize(text.replace('\\\\', ''))
        tokens = [(i.token_value, i.token_type.name) for i in tokens]
        return tokens
    except Exception as e:
        print(f"Error tokenizing text: {e}")
        return []
    

def tokenize_dataframe(df, column_name):
    with ProcessPoolExecutor() as executor:
        tokenized_functions = list(executor.map(tokenize, df[column_name].astype(str)))
    return tokenized_functions

def tokenize_dataset(dataset):
    df = Data(dataset).get_dataset()
    start = time.time()
    if 'Wild-C' not in dataset:
        df['Token'] = tokenize_dataframe(df, 'Function')
    else:
        df['Token'] = tokenize_dataframe(df, 'contents')
    print('Tokenization time: ', time.time() - start)
    print(df)
    df.drop(['Function'], axis=1, inplace=True)
    df.to_csv(f'{path}{dataset}/tokens.csv', index=False)



def get_df_timestamps(dataset, vul_only=False):
    """ Return the timestamps for a given dataset. """
    # Load vulnerable entries
    if dataset != 'Juliet':
        df = Data(dataset=dataset).get_metadata()
    else:
        return -1

    if vul_only:
        df = df[df['Vulnerable'] == 1]

    # Get timestamps
    df = df[df['Date'] != '-']
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')
    df['Date'] = df['Date'].astype("datetime64[ns]")
    df = df[['UID', 'Vulnerable', 'Date']]
    df = df.dropna()
    return df


def read_vocab(token_list):
    """
    Read the vocabulary from a list of tokens.
    """
    vocab = {}
    for tokens in token_list:
        for tok in tokens:
            # Ignore CONSTANTS and STRINGS
            if tok[1] == 'CONSTANT' or tok[1] == 'STRING':
                continue
            # Record
            if tok[0] not in vocab:
                vocab[tok[0]] = 0
            vocab[tok[0]] += 1
    return vocab


def compare_token_distribution(dataset):
    """
    Compare the token distribution of old data in comparison to new data.
    """
    # Read the data
    tokens = Data(dataset).get_tokens()
    # Append timestamps
    timestamps = get_df_timestamps(dataset, vul_only=False)
    tokens = tokens.merge(timestamps, on=['UID', 'Vulnerable'], how='inner')
    # Sort entries by time
    tokens = tokens.sort_values(by=['Date'])
    # Split entries by time
    old, new = train_test_split(tokens, test_size=0.5, shuffle=False)

    # Tokenize old and new data concurrently
    with ThreadPoolExecutor() as executor:
        old_vocab_future = executor.submit(read_vocab, old.Token.tolist())
        new_vocab_future = executor.submit(read_vocab, new.Token.tolist())

        old_vocab = old_vocab_future.result()
        new_vocab = new_vocab_future.result()

    # Normalize the vocabulary lists
    old_freq, new_freq = [], []
    for token in old_vocab:
        if token not in new_vocab:
            new_vocab[token] = 0
        old_freq.append(old_vocab[token])
        new_freq.append(new_vocab[token])

    # Calculate Jensen-Shannon Divergence
    print(f'{dataset} Immediacy: {jensenshannon(old_freq, new_freq)}')
    print()

if __name__ == '__main__':

    path = ''

    if sys.argv[1] == 'prepare':
        tokenize_dataset(sys.argv[2])
    elif sys.argv[1] == 'measure':
        compare_token_distribution(sys.argv[2])
    else:
        print(f"ERROR: Unknown command line argument: \"{sys.argv[1]}\"")

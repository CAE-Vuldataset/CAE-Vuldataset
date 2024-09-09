"""
Benchmark SOTA models on each of the selected datasets.
"""
import sys
import time
import json
import pandas as pd
from dq_analysis.datasets.data import Data
from dq_analysis.attributes.uniqueness import get_duplicate_clusters
from dq_analysis.attributes.consistency import get_consistent_dataset, get_inconsistent_clusters
from dq_analysis.svp.run import run_sota, get_random_string
from sklearn.model_selection import train_test_split


def fat_controller(experiment, dataset, model):
    """
    Control which experiment to run.
    """

    execute = {
        'benchmark': benchmark,
        'currency': currency,
        'completeness': completeness,
        'uniqueness': uniqueness,
        'consistency': consistency,
        'accuracy': accuracy,
    }
    execute[experiment](dataset, model)


def benchmark(dataset, model):
    """
    Benchmark the dataset by running 10 times using random splits.
    """

    print('Initializing: ', dataset)
    tstart = time.time()
    data = Data(dataset).get_dataset()
    train, test_data = train_test_split(data, test_size=0.2, shuffle=True)
    val, test = train_test_split(test_data, test_size=0.5, shuffle=True)
    print(time.time() - tstart, '\n#####')

    print('Running: ')
    tstart = time.time()
    run_sota(train, val, test, model, dataset, 'benchmark', save=True)
    print(time.time() - tstart, '\n#####')


def currency(dataset, model):
    """
    Investigate presence of concept drift.
    See affect of moving the test dataset further from the training dataset.
    """

    if dataset == 'Juliet':
        return  # Skip Juliet as no time data

    print('Initializing: ', dataset)
    tstart = time.time()
    # Load dataset
    if dataset != 'D2A':
        df = Data(dataset).get_dataset()
    else:
        df = get_consistent_dataset('D2A')
    # Load timestamp data
    timestamps = Data(dataset).get_metadata()
    timestamps = timestamps[timestamps['Date'] != '-']
    if dataset == 'Big-Vul':
        timestamps['Date'] = pd.to_datetime(timestamps['Date'], yearfirst=True)
    else:
        timestamps['Date'] = pd.to_datetime(timestamps['Date'], dayfirst=True)
    timestamps = timestamps.sort_values(by=['Date'])
    # Sort entries by time
    df = df.set_index('UID')
    df = df.reindex(index=timestamps['UID'])
    df = df.reset_index()
    df = df.dropna()
    df['Vulnerable'] = df['Vulnerable'].astype(int)
    # Get training set
    train, test = train_test_split(df, test_size=0.5, shuffle=False)
    train, val = train_test_split(train, test_size=0.2, shuffle=False)
    # Create test sets
    test0, test = train_test_split(test, train_size=0.2, shuffle=False)
    test1, test = train_test_split(test, train_size=0.25, shuffle=False)
    test2, test = train_test_split(test, train_size=0.33, shuffle=False)
    test3, test4 = train_test_split(test, train_size=0.5, shuffle=False)
    print(time.time() - tstart, '\n#####')

    print('Running: ')
    tstart = time.time()
    # Train initial model
    run_sota(train, val, test0, model, dataset, 'currency0')
    # Run on additional test sets
    run_sota(train, val, test1, model, dataset, 'currency1', load=True)
    run_sota(train, val, test2, model, dataset, 'currency2', load=True)
    run_sota(train, val, test3, model, dataset, 'currency3', load=True)
    run_sota(train, val, test4, model, dataset, 'currency4', load=True)
    print(time.time() - tstart, '\n#####')


def completeness(dataset, model):
    """
    Investigate the impact of incomplete code snippets.
    Simple case-control experiment in which we compare performance when a model
        is trained either using complete or incomplete code snippets.
    Use equal size training sets and common test set.
    """

    print('Initializing: ', dataset)
    tstart = time.time()
    # Get a list of incomplete examples
    trunc = pd.read_csv(f"dq_analysis/datasets/{dataset}/truncation.csv")
    # Make two unique sets
    data = Data(dataset).get_dataset()
    incomplete = data[data.UID.isin(trunc.UID)]
    if dataset == 'D2A':
        df = get_consistent_dataset('D2A')
        complete = df[~df.UID.isin(trunc.UID)]
    else:
        complete = data[~data.UID.isin(trunc.UID)]
    # Generate a common test set
    train, test = train_test_split(complete, test_size=0.1, shuffle=True)
    # Split training sets
    train, inc_train = train_test_split(train, test_size=0.5, shuffle=True)
    inc_train = inc_train.sample(n=len(inc_train)-len(incomplete))
    inc_train = pd.concat([inc_train, incomplete])
    # Get validation sets
    train, val = train_test_split(train, test_size=0.1, shuffle=True)
    inc_train, inc_val = train_test_split(inc_train, test_size=0.1, shuffle=True)

    print(f'Training set size: {len(train)}')
    print(time.time() - tstart, '\n#####')

    print('Running: ')
    tstart = time.time()
    uid = get_random_string(3)
    run_sota(train, val, test, model, dataset, f'completeness_1_{uid}')
    run_sota(inc_train, inc_val, test, model, dataset, f'completeness_0_{uid}')
    print(time.time() - tstart, '\n#####')


def uniqueness(dataset, model):
    """
    Investigate impact of within class fuzzy duplicates.
    """

    # Load the  dataset
    data = Data(dataset).get_dataset()
    clusters = get_duplicate_clusters(dataset)
    train, test_data = train_test_split(data, test_size=0.3, shuffle=True)
    val, test = train_test_split(test_data, test_size=0.33, shuffle=True)
    # Get duplicate clusters
    test_duplicates = []
    for c, x in enumerate(clusters):
        dups = [int(i) for i in x]
        for i in dups:
            if i in train.UID.tolist():
                test_duplicates += dups
                break
    test_duplicates = list(set(test_duplicates))
    test_uniq = test[~test.UID.isin(test_duplicates)]
    print(f'Data diff: {len(test)} -> {len(test_uniq)}')
    print(time.time() - tstart, '\n#####')

    print('Running: ')
    tstart = time.time()
    uid = get_random_string(3)
    # Evaluate with cross-set duplicates
    run_sota(train, val, test, model, dataset, f'uniqueness_0_{uid}', save=True)
    # Evaluate without cross-set duplicates
    run_sota(train, val, test_uniq, model, dataset, f'uniqueness_1_{uid}', load=True)
    print(time.time() - tstart, '\n#####')


def consistency(dataset, model, mode='bench'):
    """
    Investigate impact of cross-class exact duplicates.
    Remove all such duplicates from the dataset, favouring the vulnerable class.
    Determine whether the conflicting labels confuse the model.
    @param: mode: [train, test, bench]
       test = consistent train, inconsistent test
       train = inconsistent train, consistent test
       bench = all consistent
    """

    print('Initializing: ', dataset)
    tstart = time.time()
    all = Data(dataset).get_dataset()
    const = get_consistent_dataset(dataset)
    # Load inconsistent clusters
    if mode == 'bench':
        train, test_data = train_test_split(const, test_size=0.2, shuffle=True)
        val, test = train_test_split(test_data, test_size=0.5, shuffle=True)
    elif mode == 'train':
        train, test_data = train_test_split(all, test_size=0.2, shuffle=True)
        val, test = train_test_split(test_data, test_size=0.5, shuffle=True)
        # Get inconsistent clusters
        clusters = get_inconsistent_clusters(dataset, type=1, return_clusters=True)
        cross_inconsistent = []
        for c, x in enumerate(clusters):
            for id in x:
                if id in train.UID.tolist():
                    temp_label = all.at[id, 'Vulnerable']
                    # Get all inconsistent labels to the one in the training set
                    cross_inconsistent += [i for i in x if all.at[i, 'Vulnerable'] != temp_label]
        # Remove within set inconsistency
        test = test[test.UID.isin(const.UID)]
        # Remove cross set inconsistency
        test = test[~test.UID.isin(cross_inconsistent)]

    print(f'Data size: {len(all)} -> {len(train) + len(val) + len(test)}')
    print(time.time() - tstart, '\n#####')

    print('Running: ')
    tstart = time.time()
    name = 'consistency_' + mode
    run_sota(train, val, test, model, dataset, name)
    print(time.time() - tstart, '\n#####')


def accuracy(dataset, model):
    """
    Investigate model performance against a manually verified test set.
    """

    print('Initializing: ', dataset)
    tstart = time.time()
    if dataset != 'D2A':
        data = Data(dataset).get_dataset()
    else:
        data = get_consistent_dataset('D2A')
    # Load manual validated samples
    validated = pd.read_csv(f'dq_analysis/datasets/{dataset}/sample.csv')
    validated = validated.merge(data, how='left', on=['ID', 'UID', 'Vulnerable'])
    validated['Vulnerable'] = validated['Label']
    print(validated)
    # Make train/val sets
    data = data[~data.UID.isin(validated.UID)]
    print(data)
    train, test_data = train_test_split(data, test_size=0.2, shuffle=True)
    val, test = train_test_split(test_data, test_size=0.5, shuffle=True)
    print(time.time() - tstart, '\n#####')

    print('Running: ')
    tstart = time.time()
    run_sota(train, val, validated, model, dataset, 'accuracy')
    print(time.time() - tstart, '\n#####')


if __name__ == '__main__':
    experiment = sys.argv[1]
    dataset = sys.argv[2]
    model = sys.argv[3]
    fat_controller(experiment, dataset, model)
    print('Done')

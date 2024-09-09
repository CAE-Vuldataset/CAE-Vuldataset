"""
Analyse the completeness of datasets.
Consider data values which have missing information, i.e., truncation.
"""
import sys
import re
import pandas as pd
from data import Data
import time

if len(sys.argv) < 2:
    print("Usage: python dq_analysis/attributes/completeness.py [prepare/measure] <dataset>")
    exit()


def syntax_completeness(dataset):
    """
    Analyze code snippets that don't have a complete syntax.
    Either truncated at the START or END of the function.
    Return a list of IDs that have been truncated and the type of truncation.
    """

    data = Data(dataset)
    df = data.get_dataset()
    
    def check_truncation(func):
        ret = ''
        if func == '':
            return "NAN"
        
        func = re.sub(r'/\*.*?\*/', '', func, flags=re.DOTALL)
        func = re.sub(r'(?m)(^.*?)(\s*//.*)?$', r'\1', func)
        func = func.lstrip()
        
        # Check for truncation at START
        start = func.splitlines()[0]
        # Case 1: No space between function name and parameter list
        if ' (' not in start:
            space_base = 1
        # Case 2: Space between function name and parameter list
        else:
            space_base = 2
        if dataset == "BigVul":
            space_base = space_base - 1
        if start.rsplit('(', 1)[0].count(' ') < space_base and ("#include" not in start and "#define" not in start):
            ret += 'START'
    
        func = re.sub('\/\*.+?\*\/', '', func)
        
        func = re.sub(r'\s+\Z', '', func)
        if not (func.rstrip().endswith('}') or func.rstrip().endswith('};') or
                func.rstrip().endswith('#endif') or func.rstrip().endswith(');') or
                func.rstrip().endswith(')')):
            ret += 'END'

        ret = 'NONE' if ret == '' else ret
        return ret

    # Get df containing truncated entries
    df['Truncation'] = df['Function'].astype(str).apply(check_truncation)
    df = df.drop(columns=['Function'])
    df = df[df.Truncation != 'NONE']

    # pure START warnings are false positives for D2A
    #     as it handles multi-line return_types correctly.
    if dataset == 'D2A':
        df = df[df.Truncation != 'START']
        
    df.to_csv(f'{path}{dataset}/truncation.csv', index=False)


def report_completeness(dataset):
    """
    Report the measured completeness for the selected datasets,
        based on automated syntax analysis.
    """
    incomplete = pd.read_csv(f'{path}{dataset}/truncation.csv')
    data = Data(dataset).get_dataset()
    print(f'{dataset} Completeness: {(len(data) - len(incomplete))/len(data)}')
    print(f'Start Truncation: {len(incomplete[incomplete.Truncation == "START"])}')
    print(f'End Truncation: {len(incomplete[incomplete.Truncation == "END"])}')
    print(f'Both Truncation: {len(incomplete[incomplete.Truncation == "STARTEND"])}')
    print(f'Nan functions: {len(incomplete[incomplete.Truncation == "NAN"])}')
    print('-'*10)


if __name__ == '__main__':
    path ='/home/nfs/zxh2023/DataEval/data/'
    # path = '/home/nfs/zxh2023/DataEval/VulRepair/'
    if sys.argv[1] == 'prepare':
        syntax_completeness(sys.argv[2])
    elif sys.argv[1] == 'measure':
        report_completeness(sys.argv[2])
    else:
        print(f"ERROR: Unknown command line argument: \"{sys.argv[1]}\"")

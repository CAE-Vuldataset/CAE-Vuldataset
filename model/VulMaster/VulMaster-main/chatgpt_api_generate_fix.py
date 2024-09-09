import openai, csv, time, logging, difflib
import json
from tqdm import tqdm
from gpt_utils import write_text

######## Prepare Data ###########
import pandas as pd
data = pd.read_csv('./CWE_data_from_homepage.csv')
cwe_ids = data['cwe_id'].tolist()[0:97]
cwe_names = data['cwe_name'].tolist()[0:97]
cwe_descriptions = data['description'].tolist()[0:97]
codes = data['code'].tolist()[0:97]
analysis = data['analysis'].tolist()[0:97]
cwe_ids = [l.strip() for l in cwe_ids]
cwe_names = [l.strip() for l in cwe_names]
cwe_descriptions = [l.strip() for l in cwe_descriptions]
codes = [l.strip() for l in codes]
analysis[65] = 'none'
analysis = [l.strip() for l in analysis]
print()

P0 = "The code {code} contains a vulnerability of type {type}. The analysis of this vulnerable code is: {guidance}. Please generate the repaired code to address the vulnerability:"
P1 = "The code {code} suffers from {type}. Here is the analysis for the code: {guidance}. Please output the repaired code to fix the vulnerability:"
P2 = "{code} contains {type}. {guidance}. Please generate the repaired code to address the vulnerability:"
P0_examples = [ P0.format(code=codes[i], type=cwe_names[i], guidance=analysis[i]) for i in range(len(codes)) ]
P1_examples = [ P1.format(code=codes[i], type=cwe_names[i], guidance=analysis[i]) for i in range(len(codes)) ]
P2_examples = [ P2.format(code=codes[i], type=cwe_names[i], guidance=analysis[i]) for i in range(len(codes)) ]
P3_no_analysis_examples = [ P3_no_analysis.format(code=codes[i], type=cwe_names[i]) for i in range(len(codes)) ]
P4_no_cwe_examples = [ P4_no_cwe.format(type=cwe_names[i]) for i in range(len(codes)) ]

######## Query ChatGPT ###########
key = ''
openai.api_key = key



def gpt_query(gpt_input):

    completion = None
    prompt = "Please generate the repaired code to address the vulnerability. Please think twice. Let's start: "
    while completion is None:
        try:
            completion = openai.ChatCompletion.create(
                # model='gpt-3.5-turbo-0301',
                model='gpt-3.5-turbo-0613',
                # model='gpt-3.5-turbo',

                messages=[
                    {
                        "role": "system",
                        "content": "%s" % prompt,
                    },
                    {
                        "role": "user",
                        "content": gpt_input}],
            )
        except Exception as exc:
            print(exc)
            print('Failed on %s...' % idx)
            time.sleep(10)
            continue
        
        reply_content = completion.choices[0].message.content  # getting only the answer
    return reply_content


if __name__ == '__main__':
    import os
    os.mkdir ('./chatgpt_generated_P0')
    start_time = time.time()
    iteration_count = 0
    accumulate_time = 0

    for idx in tqdm(range(len(P1_examples))):
        print(idx)
        iteration_count += 1

        gpt_input = P0_examples[idx]
        # gpt_input = P1_examples[idx]
        # gpt_input = P2_examples[idx]
        if len(gpt_input.strip()) == 0:
            print('skip')
            continue
        print([gpt_input])
        gpt_output = gpt_query(gpt_input)


        print('[[[[[Generated]]]]]: \n%s' % gpt_output)
        print()

        print('-' * 100)


        end_time = time.time()
        iteration_time = end_time - start_time
        
        start_time = end_time
        accumulate_time += iteration_time
        print('Average time per a query: ', accumulate_time / iteration_count)
        write_text('./chatgpt_generated_P0' + '/%s.txt' % idx, 'w', gpt_output)
        # write_text('./chatgpt_generated_P1' + '/%s.txt' % idx, 'w', gpt_output)
        # write_text('./chatgpt_generated_P2'+'/%s.txt' % idx, 'w', gpt_output)



        print('\n\n\n')



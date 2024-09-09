import time
import sys
import pandas as pd
import numpy as np
import database as db
from collections import Counter


def filter_code(vuln_code):
	code_lines = []

	for code_line in vuln_code:
		if '//' in code_line:
			code_line = code_line[:code_line.find('//')]
		elif '/*' in code_line and '*/' in code_line:
			start_comment_index = code_line.find('/*')
			end_comment_index = code_line.find('*/')

			code_line = code_line[:start_comment_index] + code_line[end_comment_index + 2:]

		code_lines.append(code_line)

	return '\n'.join(code_lines)


def extract_vuln_code(row):
	code = np.asarray(row['code_before'].splitlines())
	vuln_lines = np.asarray(row['vuln_lines']) - 1

	if len(vuln_lines) == 0:
		return ''

	vuln_code = code[vuln_lines]

	return filter_code(vuln_code)


def extract_clean_code(row, granularity='file', output='code'):
	# Output options: code, context and all (code + context)
	if granularity == 'file':
		code = np.asarray(row['code_before'].splitlines())

		if output == 'code':
			code_lines = np.asarray(list(set(row['vuln_lines']) - set(row['noisy_lines']))) - 1
		elif output == 'context':
			code_lines = np.asarray(
				list(set(list(range(1, len(code) + 1))) - set(row['noisy_lines']) - set(row['vuln_lines']))) - 1
		elif output == 'all':
			code_lines = np.asarray(list(set(list(range(1, len(code) + 1))) - set(row['noisy_lines']))) - 1
	elif granularity == 'method':
		code = np.asarray(row['code'].splitlines())
		start_line = int(row['start_line'])

		if output == 'code':
			code_lines = np.asarray(list(set(row['method_vuln_lines']) - set(row['noisy_lines']))) - start_line
		elif output == 'context':
			method_lines = np.asarray(list(range(len(code)))) + start_line
			method_lines = method_lines.tolist()
			code_lines = np.asarray(
				list(set(method_lines) - set(row['noisy_lines']) - set(row['method_vuln_lines']))) - start_line
		elif output == 'all':
			method_lines = np.asarray(list(range(len(code)))) + start_line
			method_lines = method_lines.tolist()
			code_lines = np.asarray(
				list(set(method_lines) - set(row['noisy_lines']))) - start_line

	if len(code_lines) == 0:
		return ''

	code = code[code_lines]

	return filter_code(code)


def extract_non_vuln_code_file(row, context_dict):
	code = np.asarray(row['code_before'].splitlines())

	code_lines = np.asarray(list(set(list(range(1, len(code) + 1))) - set(row['noisy_lines']) -
								 set(context_dict[row['file_change_id']]))) - 1

	if len(code_lines) == 0:
		return ''

	code = code[code_lines]

	return filter_code(code)


def extract_non_vuln_code_method(row):
	code = np.asarray(row['code'].splitlines())
	start_line = int(row['start_line'])

	method_lines = np.asarray(list(range(len(code)))) + start_line
	method_lines = method_lines.tolist()

	code_lines = np.asarray(list(set(method_lines) - set(row['noisy_lines']) -
								 set(row['context_lines']))) - start_line

	if len(code_lines) == 0:
		return ''

	code = code[code_lines]

	return filter_code(code)


def extract_surrounding_context_code(row, granularity):
	if granularity == 'file':
		code = np.asarray(row['code_before'].splitlines())
		vuln_lines = np.asarray(list(set(row['surrounding_context']) - set(row['noisy_lines']))) - 1
	elif granularity == 'method':
		code = np.asarray(row['code'].splitlines())
		start_line = int(row['start_line'])
		vuln_lines = np.asarray(list(set(row['surrounding_context']) - set(row['noisy_lines']))) - start_line

	if len(vuln_lines) == 0:
		return ''

	vuln_code = code[vuln_lines]

	return filter_code(vuln_code)


def extract_context_code_file(row, context_dict):
	if len(context_dict[row['file_change_id']]) == 0:
		return ''

	code = np.asarray(row['code_before'].splitlines())

	vuln_lines = np.asarray(list(set(context_dict[row['file_change_id']]) - set(row['noisy_lines']))) - 1

	if len(vuln_lines) == 0:
		return ''

	vuln_code = code[vuln_lines]

	return filter_code(vuln_code)


def extract_method_vuln_code(row):
	code = np.asarray(row['code'].splitlines())
	start_line = int(row['start_line'])
	vuln_lines = np.asarray(row['method_vuln_lines']) - start_line

	if len(vuln_lines) == 0:
		return ''

	vuln_code = code[vuln_lines]

	return filter_code(vuln_code)


def extract_context_code_method(row):
	code = np.asarray(row['code'].splitlines())
	start_line = int(row['start_line'])
	context_lines = np.asarray(list(set(row['context_lines']) - set(row['noisy_lines']))) - start_line

	if len(context_lines) == 0:
		return ''

	context_code = code[context_lines]

	return filter_code(context_code)


def extract_context_code_file_wo_vuln(row, context_dict):
	if len(context_dict[row['file_change_id']]) == 0:
		return ''

	code = np.asarray(row['code_before'].splitlines())

	vuln_lines = np.asarray(
		list(set(context_dict[row['file_change_id']]) - set(row['vuln_lines']) - set(row['noisy_lines']))) - 1

	if len(vuln_lines) == 0:
		return ''

	vuln_code = code[vuln_lines]

	return filter_code(vuln_code)


def extract_context_code_method_wo_vuln(row):
	code = np.asarray(row['code'].splitlines())
	start_line = int(row['start_line'])
	context_lines = np.asarray(
		list(set(row['context_lines']) - set(row['method_vuln_lines']) - set(row['noisy_lines']))) - start_line

	if len(context_lines) == 0:
		return ''

	context_code = code[context_lines]

	return filter_code(context_code)


def extract_context_code_method_wo_vuln_lines(row):

	start_line = int(row['start_line'])
	context_lines = np.asarray(list(set(row['context_lines']) - set(row['method_vuln_lines']) - set(row['noisy_lines']))) - start_line

	if len(context_lines) == 0:
		return []

	return context_lines


def extract_surrounding_context_code_wo_vuln(row, granularity):
	if granularity == 'file':
		code = np.asarray(row['code_before'].splitlines())

		vuln_lines = np.asarray(
			list(set(row['surrounding_context']) - set(row['vuln_lines']) - set(row['noisy_lines']))) - 1

	elif granularity == 'method':
		code = np.asarray(row['code'].splitlines())
		start_line = int(row['start_line'])
		vuln_lines = np.asarray(list(
			set(row['surrounding_context']) - set(row['method_vuln_lines']) - set(row['noisy_lines']))) - start_line

	if len(vuln_lines) == 0:
		return ''

	vuln_code = code[vuln_lines]

	return filter_code(vuln_code)


def extract_left_right_context(vuln_lines, context_lines):
	start_line, end_line = vuln_lines[0], vuln_lines[-1]
	context_lines = np.asarray(list(set(context_lines) - set(vuln_lines)))

	return context_lines[context_lines < end_line], context_lines[context_lines > start_line]


def create_fold(df, key, folds):
	sizes = []
	fold_sum = 0

	if type(folds) is list:

		for i in range(len(folds)):
			if i == len(folds) - 1:
				sizes.append(len(df) - 1)
			else:
				sizes.append(int(len(df) * folds[i]) + fold_sum)
				fold_sum += int(len(df) * folds[i])
	else:

		# print("Here")

		size_per_fold = int(len(df) / folds)

		for i in range(folds):
			if i == folds - 1:
				sizes.append(len(df) - 1)
			else:
				sizes.append(size_per_fold + fold_sum)
				fold_sum += size_per_fold

	tmp_df = df.copy()
	tmp_df['row_index'] = list(range(len(df)))
	tmp_df = tmp_df.rename(columns={key: 'key'})

	tmp_df['fold'] = 0

	for i, size in enumerate(sizes):

		if i == 0:
			start_index = 0
		else:
			start_index = sizes[i - 1] + 1

		end_index = size
		
		tmp_df.loc[(start_index <= tmp_df['row_index']) & (tmp_df['row_index'] <= end_index), 'fold'] = i

	fold_map = tmp_df[['key', 'fold']].copy()
	fold_map['key'] = fold_map['key'].astype(str)
	fold_map['fold'] = fold_map['fold'].astype(int)
	
	return fold_map


def change_whole_method(row):
	start_line, end_line = int(row['start_line']), int(row['end_line'])
	cur_vuln_lines = row['method_vuln_lines']
	noisy_lines = row['noisy_lines']

	code = row['code'].splitlines()
	line_no = 0

	while not ")" in code[line_no].strip():
		line_no += 1

	method_lines = list(range(start_line, end_line + 1))

	unchanged_lines = list(set(method_lines) - (set(cur_vuln_lines).union(set(noisy_lines))))
	
	if len(unchanged_lines) <= 2 + line_no:
		return "True"

	return "False"


def extract_context_scope(row, granularity, scope_size=5):
	context_lines = []

	if granularity == 'file':
		vuln_lines = row['vuln_lines']
		nloc = row['nloc_new']
		context_lines = vuln_lines.copy()

		for line in vuln_lines:
			start_scope = line - scope_size

			if start_scope < 1:
				start_scope = 1

			end_scope = line + scope_size

			if end_scope > nloc:
				end_scope = nloc

			context_lines.extend([line_index for line_index in range(start_scope, end_scope + 1)])

	elif granularity == 'method':
		vuln_lines = row['method_vuln_lines']
		context_lines = vuln_lines.copy()
		start_line = int(row['start_line'])
		end_line = int(row['end_line'])

		for line in vuln_lines:
			start_scope = line - scope_size

			if start_scope < start_line:
				start_scope = start_line

			end_scope = line + scope_size

			if end_scope > end_line:
				end_scope = end_line

			context_lines.extend([line_index for line_index in range(start_scope, end_scope + 1)])

	return sorted(list(set(context_lines)))


def extract_random_context(row, granularity):
	if granularity == 'file':
		code = np.asarray(row['code_before'].splitlines())

		code_lines = np.asarray(
			list(set(list(range(1, len(code) + 1))) - set(row['noisy_lines']) - set(row['vuln_lines']))) - 1

		vuln_lines_len = len(row['vuln_lines'])
		if vuln_lines_len > len(code_lines):
			vuln_lines_len = len(code_lines)

		code_lines = np.random.RandomState(vuln_lines_len).choice(code_lines, vuln_lines_len, replace=False)

	elif granularity == 'method':
		code = np.asarray(row['code'].splitlines())
		start_line = int(row['start_line'])

		method_lines = np.asarray(list(range(len(code)))) + start_line
		method_lines = method_lines.tolist()
		code_lines = np.asarray(
			list(set(method_lines) - set(row['noisy_lines']) - set(row['method_vuln_lines']))) - start_line

		vuln_lines_len = len(row['method_vuln_lines'])
		if vuln_lines_len > len(code_lines):
			vuln_lines_len = len(code_lines)

		code_lines = np.random.RandomState(vuln_lines_len).choice(code_lines, vuln_lines_len, replace=False)

	if len(code_lines) == 0:
		return ''

	code = code[code_lines]

	return filter_code(code)


def method_size(row):
	code = np.asarray(row['code'].splitlines())
	start_line = int(row['start_line'])
	method_lines = np.asarray(list(range(len(code)))) + start_line
	method_lines = method_lines.tolist()
	code_lines = np.asarray(list(set(method_lines) - set(row['noisy_lines']))) - start_line
	return len(code_lines)

df_method = pd.read_parquet('Data/combined_df_method.parquet')

df_method[['author_date', 'committer_date']] = df_method[['author_date', 'committer_date']].apply(
	lambda r: pd.to_datetime(r, infer_datetime_format=True))

df_method['nloc_method'] = df_method[['code', 'start_line', 'noisy_lines']].apply(
	lambda r: method_size(r), axis=1)

df_method['method_ratio'] = df_method[['method_vuln_lines', 'nloc_method']].apply(
    lambda r: len(r['method_vuln_lines']) / r['nloc_method'] * 1.0, axis=1)

df_method['whole_method_change'] = df_method[
	['code', 'start_line', 'end_line', 'method_vuln_lines', 'noisy_lines']].apply(
	lambda r: change_whole_method(r), axis=1
)

# Program slicing w/o vuln lines
df_method['ps_only'] = df_method[['code', 'context_lines', 'start_line', 'method_vuln_lines', 'noisy_lines']].apply(
	lambda r: extract_context_code_method_wo_vuln(r), axis=1)

df_method['ps_only_ratio'] = df_method[['ps_only', 'nloc_method']].apply(
	lambda r: len(r['ps_only']) / r['nloc_method'], axis=1)

df_method = df_method[(df_method['method_ratio'] < 1.0) &
					  (df_method['whole_method_change'] == 'False') & (df_method['ps_only_ratio'] > 0)]

df_method = df_method.iloc[np.random.RandomState(42).permutation(len(df_method))].reset_index(drop=True)

n_folds = 10

method_map = create_fold(df_method, 'method_change_id', folds=n_folds)

method_map.to_csv('Data/method_map.csv', index=False)

cvss_cols = ['cvss2_confidentiality_impact', 'cvss2_integrity_impact', 'cvss2_availability_impact',
			 'cvss2_access_vector', 'cvss2_access_complexity', 'cvss2_authentication', 'severity']

# Whole method
selected_cols = ['method_change_id', 'code', 'noisy_lines', 'start_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp['filtered_code'] = df_tmp[['code', 'noisy_lines', 'start_line']].apply(
	lambda r: extract_clean_code(r, 'method', 'all'), axis=1)
df_tmp = df_tmp.drop(columns=['code', 'noisy_lines', 'start_line'])
df_tmp = df_tmp.rename(columns={'method_change_id': 'key', 'filtered_code': 'code'}).reset_index(drop=True)
print(len(df_tmp), df_tmp.columns)
df_tmp.to_parquet('Data/method_whole.parquet', index=False)

# Vuln lines without context in methods
selected_cols = ['method_change_id', 'code', 'method_vuln_lines', 'start_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp['vuln_code'] = df_tmp[['code', 'method_vuln_lines', 'start_line']].apply(lambda r: extract_method_vuln_code(r), axis=1)
df_tmp = df_tmp.drop(columns=['code', 'method_vuln_lines', 'start_line'])
df_tmp = df_tmp.rename(columns={'vuln_code': 'code', 'method_change_id': 'key'}).reset_index(drop=True)
print(len(df_tmp), df_tmp.columns)
df_tmp.to_parquet('Data/method_lines_without_context.parquet', index=False)

# Vuln lines with surrounding context (consecutive lines before and after the vuln. lines) in methods
scope_size = 6

selected_cols = ['method_change_id', 'code', 'method_vuln_lines', 'noisy_lines', 'start_line', 'end_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()

df_tmp['surrounding_context'] = df_tmp[['method_vuln_lines', 'start_line', 'end_line']].apply(
	lambda r: extract_context_scope(r, granularity='method', scope_size=scope_size), axis=1)

df_tmp['context_code'] = df_tmp[['code', 'surrounding_context', 'noisy_lines', 'start_line']].apply(
	lambda r: extract_surrounding_context_code(r, granularity='method'), axis=1)

df_tmp = df_tmp.drop(columns=['code', 'method_vuln_lines', 'start_line', 'end_line', 'surrounding_context', 'noisy_lines'])
df_tmp = df_tmp.rename(columns={'context_code': 'code', 'method_change_id': 'key'}).reset_index(drop=True)

print(len(df_tmp), df_tmp.columns)
df_tmp.to_parquet('Data/method_lines_with_surrounding_context.parquet', index=False)


# Vuln lines with context in methods
selected_cols = ['method_change_id', 'code', 'context_lines', 'start_line', 'noisy_lines']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp['vuln_code'] = df_tmp[['code', 'context_lines', 'start_line', 'noisy_lines']].apply(lambda r: extract_context_code_method(r), axis=1)
df_tmp = df_tmp.drop(columns=['code', 'context_lines', 'start_line', 'noisy_lines'])
df_tmp = df_tmp.rename(columns={'vuln_code': 'code', 'method_change_id': 'key'}).reset_index(drop=True)
print(len(df_tmp), df_tmp.columns)
df_tmp.to_parquet('Data/method_lines_with_all_context.parquet', index=False)

##########################

# Context only in methods
selected_cols = ['method_change_id', 'code', 'noisy_lines', 'method_vuln_lines', 'start_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp['filtered_code'] = df_tmp[['code', 'noisy_lines', 'method_vuln_lines', 'start_line']].apply(
	lambda r: extract_clean_code(r, 'method', 'context'), axis=1)

df_tmp = df_tmp.drop(columns=['code', 'noisy_lines', 'method_vuln_lines', 'start_line'])
df_tmp = df_tmp.rename(columns={'method_change_id': 'key', 'filtered_code': 'code'}).reset_index(drop=True)
print(len(df_tmp), df_tmp.columns)
df_tmp.to_parquet('Data/method_context_only.parquet', index=False)

###########################

# Non vuln lines in methods (all scope - program slicing scope)
selected_cols = ['method_change_id', 'code', 'context_lines', 'start_line', 'noisy_lines']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp['vuln_code'] = df_tmp[['code', 'context_lines', 'start_line', 'noisy_lines']].apply(
	lambda r: extract_non_vuln_code_method(r), axis=1)
df_tmp = df_tmp.drop(columns=['code', 'context_lines', 'start_line', 'noisy_lines'])
df_tmp = df_tmp.rename(columns={'vuln_code': 'code', 'method_change_id': 'key'}).reset_index(drop=True)
print(len(df_tmp), df_tmp.columns)
df_tmp.to_parquet('Data/method_non_vuln.parquet', index=False)

###########################

# Vuln lines with surrounding context (consecutive lines before and after the vuln. lines) in methods
scope_size = 6

selected_cols = ['method_change_id', 'code', 'method_vuln_lines', 'start_line', 'end_line', 'noisy_lines']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()

df_tmp['surrounding_context'] = df_tmp[['method_vuln_lines', 'start_line', 'end_line']].apply(
	lambda r: extract_context_scope(r, granularity='method', scope_size=scope_size), axis=1)

df_tmp['context_code'] = df_tmp[['code', 'surrounding_context', 'start_line', 'method_vuln_lines', 'noisy_lines']].apply(
	lambda r: extract_surrounding_context_code_wo_vuln(r, granularity='method'), axis=1)

df_tmp = df_tmp.drop(columns=['code', 'method_vuln_lines', 'start_line', 'end_line', 'noisy_lines', 'surrounding_context'])
df_tmp = df_tmp.rename(columns={'context_code': 'code', 'method_change_id': 'key'}).reset_index(drop=True)

print(len(df_tmp), df_tmp.columns)
df_tmp.to_parquet('Data/method_surrounding_only.parquet', index=False)

###########################

# Vuln lines with context in methods
selected_cols = ['method_change_id', 'code', 'context_lines', 'start_line', 'method_vuln_lines', 'noisy_lines']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()

df_tmp['context_code'] = df_tmp[['code', 'context_lines', 'start_line', 'method_vuln_lines', 'noisy_lines']].apply(
	lambda r: extract_context_code_method_wo_vuln(r), axis=1)

df_tmp = df_tmp.drop(columns=['code', 'context_lines', 'start_line', 'method_vuln_lines', 'noisy_lines'])
df_tmp = df_tmp.rename(columns={'context_code': 'code', 'method_change_id': 'key'}).reset_index(drop=True)
print(len(df_tmp), df_tmp.columns)
df_tmp.to_parquet('Data/method_slicing_only.parquet', index=False)

###########################

# Context only with the same size as the vulnerable statements in methods
selected_cols = ['method_change_id', 'code', 'noisy_lines', 'method_vuln_lines', 'start_line']
selected_cols.extend(cvss_cols)
df_tmp = df_method[selected_cols].copy()
df_tmp['filtered_code'] = df_tmp[['code', 'noisy_lines', 'method_vuln_lines', 'start_line']].apply(
	lambda r: extract_random_context(r, 'method'), axis=1)

df_tmp = df_tmp.drop(columns=['code', 'noisy_lines', 'method_vuln_lines', 'start_line'])
df_tmp = df_tmp.rename(columns={'method_change_id': 'key', 'filtered_code': 'code'}).reset_index(drop=True)
print(len(df_tmp), df_tmp.columns)
df_tmp.to_parquet('Data/method_random_context.parquet', index=False)

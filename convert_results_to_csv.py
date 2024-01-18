__author__ = "qiao"

"""
Automatically evaluate the relation of LLM-classification and human eligibility annotation.
"""

import json
from nltk.tokenize import sent_tokenize
import pandas as pd
import re
import sys

def summarize_matching(
		matching: dict,
) -> list:
	"""
	Summarize the matching json dict between criteria and the patient sents

	Args:
		matching (Dict{'inclusion': Dict{Str(criteria): [Str(eligibility), List[Int(sentence_id)], Str(explanation)}, 'exclusion': Dict{Str(criteria): [Str(eligibility), List[Int(sentence_id)], Str(explanation)]}}): output of ChatGPT.
	
	Return:
			"inclusion",
			"included",
			"not included",
			"inclusion not applicable",
			"no inclusion information",
			"exclusion",
			"excluded",
			"not excluded",
			"exclusion not applicable",
			"no exclusion information",
	"""
	# count only the valid ones
	included = 0
	not_inc = 0
	na_inc = 0
	no_info_inc = 0

	excluded = 0
	not_exc = 0
	na_exc = 0
	no_info_exc = 0
	
	# first count inclusions
	for criteria, info in matching["inclusion"].items():
		if len(info) == 1:
			info = info[0]

		if len(info) != 4:
			continue

		if info[2] == "included":
			included += 1	
		elif info[2] == "not included":
			not_inc += 1
		elif info[2] == "not applicable":
			na_inc += 1
		elif info[2] == "not enough information":
			no_info_inc += 1
	
	# then count exclusions
	for criteria, info in matching["exclusion"].items():
		if len(info) == 1:
			info = info[0]

		if len(info) != 4:
			continue

		if info[2] == "excluded":
			excluded += 1	
		elif info[2] == "not excluded":
			not_exc += 1
		elif info[2] == "not applicable":
			na_exc += 1
		elif info[2] == "not enough information":
			no_info_exc += 1

	
	return [
			len(matching["inclusion"]),
			included,
			not_inc,
			na_inc,
			no_info_inc,
			len(matching["exclusion"]),
			excluded,
			not_exc,
			na_exc,
			no_info_exc,
	]


def extract_self_assess_score(assessments):
	score_list = []

	for assessment in assessments:
		pattern = r"\[.*?(\d+)\]"
		matches = re.findall(pattern, assessment)

		if matches:
			score_list.append(int(matches[-1]))
	
	if len(score_list) == 0:
		return 50
	else:
		return sum(score_list) / len(score_list)


def extract_self_releli_score(assessments):
	assessments = [assessments]

	rel_list = []
	eli_list = []

	for assessment in assessments:
		# first extract the relevance value
		r_value = re.search(r"R\s*=\s*([-]?\d+)", assessment)

		if r_value:
			r_value = float(r_value.group(1))
			rel_list.append(r_value)

		# then extract the eligibility value
		e_value = re.search(r"E\s*=\s*([-]?\d+)", assessment)

		if e_value:
			e_value = float(e_value.group(1))
			eli_list.append(e_value)

	if len(rel_list) > 0:
		avg_rel = sum(rel_list) / len(rel_list)
	else:
		avg_rel = 50
	
	if len(eli_list) > 0:
		avg_eli = sum(eli_list) / len(eli_list)
	else:
		avg_eli = 0
	
	return avg_rel, avg_eli


if __name__ == "__main__":
	# specify the dataset split
	split = sys.argv[1]
	model = sys.argv[2]

	dataset = json.load(open(f"datasets/trial_{split}.json"))
	results = json.load(open(f"results/trial_{split}_{model}_matching_results.json"))
	aggreg = json.load(open(f"results/trial_{split}_{model}_aggregation_results.json")) 

	# record number of patient-trial pairs planned to do
	num_all = 0

	for entry in dataset:
		# entry is at patient-level
		for label in ["0", "1", "2"]:
			num_all += len(entry.get(label, []))

	# this df contains counts that will be used for downstream purposes
	df_cols = [
		"patient id",
		"trial id",
		"label",
		"inclusion",
		"included",
		"not included",
		"inclusion not applicable",
		"no inclusion information",
		"exclusion",
		"excluded",
		"not excluded",
		"exclusion not applicable",
		"no exclusion information",
		"relevance",
		"eligibility"
	]
	df_json = {col: [] for col in df_cols} 
	
	# iterature over cached API result
	for patient_id, label2trial2output in results.items():

		# at the label level
		for label, trial2output in label2trial2output.items():
			# if no preds made for this label, continue to next
			if len(trial2output) == 0: continue

			# collecting the whole trial_matcing.csv
			for trial_id, output in trial2output.items():
				# now this is the level of patient-trial, and output contains the LLM annotations 
				feature = summarize_matching(output)
				
				# then extract the self rel, releli score
				if patient_id in aggreg and trial_id in aggreg[patient_id]:
					rel, eli = extract_self_releli_score(aggreg[patient_id][trial_id])
				else:
					rel, eli = float("nan"), float("nan")
				
				cols = [patient_id, trial_id, int(label)] + feature + [rel, eli]

				for col, df_col in zip(cols, df_cols):
					df_json[df_col].append(col)

	df = pd.DataFrame(df_json)
	df.to_csv(f"results/trial_matching_{split}_{model}.csv")

__author__ = "qiao"

"""
TrialGPT main functions.
"""

import json
from nltk.tokenize import sent_tokenize
import time

from config import config 

import openai
openai.api_type = config["api_type"]
openai.api_base = config["api_base"] 
openai.api_version = config["api_version"]
openai.api_key = config["api_key"] 

def print_trial(
	trial_info: dict,
	inc_exc: str,
) -> str:
	"""Given a dict of trial information, returns a string of trial."""
	
	trial = f"Title: {trial_info['brief_title']}\n"
	trial += f"Target diseases: {', '.join(trial_info['diseases_list'])}\n"
	trial += f"Interventions: {', '.join(trial_info['drugs_list'])}\n"
	trial += f"Summary: {trial_info['brief_summary']}\n"

	trial += "Inclusion criteria: %s\n" % trial_info['inclusion_criteria'].replace('\n\n', '\n')
	trial += "Exclusion criteria: %s\n" % trial_info['exclusion_criteria'].replace('\n\n', '\n')

	return trial


def get_matching_prompt(
	trial_info: dict,
	inc_exc: str,
	patient: str,
) -> str:
	"""Output the prompt."""
	prompt = f"You are a helpful assistant for clinical trial recruitment. Your task is to compare a given patient note and the {inc_exc} criteria of a clinical trial to determine the patient's eligibility at the criterion level.\n"

	if inc_exc == "inclusion":
		prompt += "The factors that allow someone to participate in a clinical study are called inclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"
	
	elif inc_exc == "exclusion":
		prompt += "The factors that disqualify someone from participating are called exclusion criteria. They are based on characteristics such as age, gender, the type and stage of a disease, previous treatment history, and other medical conditions.\n"

	prompt += f"You should check the {inc_exc} criteria one-by-one, and output the following four elements for each criterion:\n"
	prompt += f"\tElement 1. For each {inc_exc} criterion, think step-by-step and generate your reasoning process: First, judge whether the criterion is not applicable (not very common), where the patient does not meet the premise of the criterion. Then, check if the patient note contains direct evidence. If so, judge whether the patient meets or does not meet the criterion. If there is no direct evidence, try to infer from existing evidence, and answer one question: If the criterion is true, is it possible that a good patient note will miss such information? If impossible, then you can assume that the criterion is not true. Otherwise, there is not enough information.\n"
	prompt += f"\tElement 2. If there is relevant information, you must generate a list of relevant sentence IDs in the patient note. If there is no relevant information, you must annotate an empty list.\n" 
	prompt += f"\tElement 3. Classify the patient eligibility for this specific {inc_exc} criterion: "
	
	if inc_exc == "inclusion":
		prompt += 'the label must be chosen from {"not applicable", "not enough information", "included", "not included"}. "not applicable" should only be used for criteria that are not applicable to the patient. "not enough information" should be used where the patient note does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. "included" denotes that the patient meets the inclusion criterion, while "not included" means the reverse.\n'
	elif inc_exc == "exclusion":
		prompt += 'the label must be chosen from {"not applicable", "not enough information", "excluded", "not excluded"}. "not applicable" should only be used for criteria that are not applicable to the patient. "not enough information" should be used where the patient note does not contain sufficient information for making the classification. Try to use as less "not enough information" as possible because if the note does not mention a medically important fact, you can assume that the fact is not true for the patient. "excluded" denotes that the patient meets the exclusion criterion and should be excluded in the trial, while "not excluded" means the reverse.\n'
	
	prompt += f'\tElement 4. Indicate whether the eligibility is classified based on direct evidence or inferred evidence. This reasoning type must be either "direct" or "inferred".\n'

	prompt += "You should output only a JSON dict exactly formatted as: ```dict{str(%s_criterion): list[str(element_1_step_by_step_reasoning), list[int(element_2_sentence_id)], str(element_3_eligibility_label), str(element_4_reasoning_type)]}\n```" % inc_exc
	
	prompt += f"Here is the patient note, each sentence is led by a sentence_id:\n{patient}\n" 
	prompt += f"Here is the clinical trial:\n{print_trial(trial_info, inc_exc)}\n"
	prompt += f"Plain JSON output:\n"

	return prompt


def trialgpt_matching(trial: dict, patient: str, model: str):
	results = {}

	# doing inclusions and exclusions in separate prompts
	for inc_exc in ["inclusion", "exclusion"]:
		prompt = get_matching_prompt(trial, inc_exc, patient)

		# some trials don't have inclusions / exclusions
		if not prompt:
			results[inc_exc] = {}
			continue
		
		# sleep 1 second for the rate limit
		time.sleep(1)
		completion = openai.ChatCompletion.create(
				engine=model,
				messages=[{"role": "user", "content": prompt}],
				temperature=0,
		)
		message = completion.choices[0].message["content"].strip()
		results[inc_exc] = json.loads(message)

	return results


def convert_criteria_pred_to_string(
		prediction: dict,
) -> str:
	"""Given the TrialGPT prediction, output the linear string of the criteria."""
	output = ""

	for inc_exc in ["inclusion", "exclusion"]:
		for idx, info in enumerate(prediction[inc_exc].items()):
			criterion, preds = info

			if len(preds) != 3:
				continue

			output += f"{inc_exc} criterion {idx}: {criterion}\n"
			output += f"\tQuestion: How is the patient relevant to this criterion?\n"
			output += f"\tPrediction: {preds[0]}\n"
			output += f"\tQuestion: What sentences are relevant?\n"
			output += f"\tPrediction: {preds[1]}\n"
			output += f"\tQuestion: What is the eligibility of the patient for this criterion?\n"
			output += f"\tPrediction: {preds[2]}\n"
	
	return output


def convert_pred_to_prompt(
		patient: str,
		pred: dict,
		trial_info: dict,
) -> str:
	"""Convert the prediction to a prompt string."""
	# first get the patient string 
	sents = sent_tokenize(patient)
	sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
	patient = "\n".join(sents)

	# then get the trial string
	trial = f"Title: {trial_info['brief_title']}\n"
	trial += f"Target conditions: {', '.join(trial_info['diseases_list'])}\n"
	trial += f"Summary: {trial_info['brief_summary']}"

	# then get the prediction strings
	pred = convert_criteria_pred_to_string(pred)

	# construct the prompt
	prompt = "Hello. You are a helpful assistant for clinical trial recruitment. You will be given a patient note, a clinical trial, and the patient eligibility predictions for each criterion.\n"
	prompt += "Here is the patient note:\n"
	prompt += patient + "\n"
	prompt += "Here is the clinical trial description:\n"
	prompt += trial + "\n"
	prompt += "Here are the criterion-levle eligibility prediction:\n"
	prompt += pred + "\n"
	prompt += "Your task is to output two scores, a relevance score (R) and an eligibility score (E), between the patient and the clincal trial.\n"
	prompt += "First explain the consideration for determining patient-trial relevance. Predict the relevance score R (0~100), which represents the overall relevance between the patient and the clinical trial. R=0 denotes the patient is totally irrelevant to the clinical trial, and R=100 denotes the patient is exactly relevant to the clinical trial.\n"
	prompt += "Then explain the consideration for determining patient-trial eligibility. Predict the eligibility score E (-R~R), which represents the patient's eligibility to the clinical trial. Note that -R <= E <= R (the absolute value of eligibility cannot be higher than the relevance), where E=-R denotes that the patient is ineligible (not included by any inclusion criteria, or excluded by all exclusion criteria), E=R denotes that the patient is eligible (included by all inclusion criteria, and not excluded by any exclusion criteria), E=0 denotes the patient is neutral (i.e., no relevant information for all inclusion and exclusion criteria).\n"
	prompt += "Finally, you should always repeat the R and E scores in the last line (first R, then E) by `R=?, E=?`, e.g., `R=75, E=-50`."

	return prompt


def trialgpt_aggregation(patient: str, trial_results: dict, trial_info: dict):
	prompt = convert_pred_to_prompt(
			patient,
			trial_results,
			trial_info
	)   

	completion = openai.ChatCompletion.create(
			engine="gpt-4",
			messages=[
				{"role": "user", "content": prompt},
			],
			temperature=0.0,
	)

	result = completion.choices[0].message["content"].strip() 

	return result

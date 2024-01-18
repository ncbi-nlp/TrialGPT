__author__ = "qiao"

"""
Using GPT to aggregate the scores by itself.
"""

import json
import os
import sys
import time

from TrialGPT import trialgpt_aggregation

if __name__ == "__main__":
	# first loading the trial2info dict
	trial2info = json.load(open("datasets/trial2info.json"))

	# saving and cahcing
	split = sys.argv[1]
	model = sys.argv[2]
	output_path = f"results/trial_{split}_{model}_aggregation_results.json"

	if os.path.exists(output_path):
		output = json.load(open(output_path))
	else:
		output = {}
	
	# cohort-level, 3 cohorts
	# TrialGPT criterion-level predictions
	results = json.load(open(f"results/trial_{split}_{model}_matching_results.json"))

	# original patient information
	dataset = json.load(open(f"datasets/trial_{split}.json"))
	pid2note = {entry["patient_id"]: entry["patient"] for entry in dataset}

	# patient-level, 184 patients in total
	for patient_id, info in results.items():
		# get the patient note
		patient = pid2note[patient_id]

		if patient_id not in output:
			output[patient_id] = {}
		
		# label-level, 3 label / patient
		for label, trials in info.items():
				
			# trial-level, at most 50 trial / label
			for trial_id, trial_results in trials.items():
				# specific trial information
				trial_info = trial2info[trial_id]	

				if trial_id in output[patient_id]:
					continue

				result = trialgpt_aggregation(patient, trial_results, trial_info, model)
				output[patient_id][trial_id] = result 

				with open(output_path, "w") as f:
					json.dump(output, f, indent=4)

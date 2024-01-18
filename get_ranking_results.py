__author__ = "qiao"

"""
Get the reranking results from the trial matching csv 
"""

from collections import Counter
import pandas as pd
import json
import random
random.seed(2023)
import sys

from sklearn.metrics import ndcg_score
from sklearn.metrics import roc_curve, auc


def get_metrics(labels, scores):
	"""Input a list of labels and a list of scores, output the precision at 10 and MRR"""
	label_score = zip(labels, scores)
	label_score = sorted(label_score, key=lambda x: -x[1])

	top_scores = [score for _, score in label_score[:10]]
	top_labels = [label for label, _ in label_score[:10]]

	if len(top_scores) == 10:
		if top_scores[0] == top_scores[9]:
			# all ties
			labels = [label for label, score in label_score if score == top_scores[0]]
			prec = sum(labels) / (2 * len(labels))
		else:
			prec = sum(top_labels) / 20
	else:
		prec = sum(top_labels) / (2 * len(top_labels))
	
	mrr = 0
	for rank, label in enumerate(top_labels):
		if label > 0:
			mrr = 1 / (rank + 1)
			break
	
	return prec, mrr


if __name__ == "__main__":
	# first we need to combine the output csv files
	df_list = []

	model = sys.argv[1] 
	
	for cohort in ["sigir", "2021", "2022"]:
		df = pd.read_csv(f"results/trial_matching_{cohort}_{model}.csv")
		df["patient id"] = df["patient id"].apply(lambda x: cohort + " " + str(x))
		df_list.append(df)

	df = pd.concat(df_list)

	num_rows = len(df)
	random_scores = [random.uniform(0, 1) for _ in range(num_rows)]

	df["inclusion"] = df["inclusion"] - df["inclusion not applicable"]
	df["exclusion"] = df["exclusion"] - df["exclusion not applicable"]

	df["random"] = random_scores
	df["% inc"] = df["included"] / df["inclusion"]
	df["% not inc"] = - df["not included"] / df["inclusion"]
	df["bool not inc"] = - (df["not included"] > 0).astype(float)

	df["% exc"] = - df["excluded"] / df["exclusion"]
	df["% not exc"] = df["not excluded"] / df["exclusion"]
	df["bool exc"] = - (df["excluded"] > 0).astype(float)

	df["comb"] = df["% inc"] + df["bool exc"] + df["bool not inc"]  + (df["relevance"] + df["eligibility"]) / 100

	df = df.dropna()

	patient_index = df.groupby("patient id")

	score_names = ["comb", "% inc", "% not inc", "% exc", "% not exc", "bool not inc", "bool exc", "random", "eligibility", "relevance"]
	ndcg_list = {score_name: [] for score_name in score_names}
	prec_list = {score_name: [] for score_name in score_names}
	mrr_list = {score_name: [] for score_name in score_names}
	auc_list = {score_name: [] for score_name in score_names}

	for patient_id, patient_data in patient_index:
		labels = patient_data["label"].tolist()
		if len(Counter(labels)) == 1:
			continue
		
		# if there is only one label, just continue
		if len(set(labels)) <= 1: 
			continue
		
		for score_name in score_names:
			scores = patient_data[score_name].tolist()	

			# first get ndcg
			ndcg = ndcg_score([labels], [scores], k=10)
			ndcg_list[score_name].append(ndcg)

			prec, mrr = get_metrics(labels, scores)
			prec_list[score_name].append(prec)
			mrr_list[score_name].append(mrr)

			# then get auc
			if "sigir" in patient_id:
				continue

			filt_labels = []
			filt_scores = []

			for label, score in zip(labels, scores):
				if int(label) > 0:
					filt_labels.append(label)
					filt_scores.append(-score)

			if len(set(filt_labels)) == 1:
				continue

			fpr, tpr, thr = roc_curve(filt_labels, filt_scores, pos_label=1)
			auc_list[score_name].append(auc(fpr, tpr))

	print("Ranking NDCG@10")	
	for score_name in score_names:
		ndcgs = ndcg_list[score_name]

		print(score_name, sum(ndcgs) / len(ndcgs))

	print("Ranking Prec@10")	
	for score_name in score_names:
		precs = prec_list[score_name]

		print(score_name, sum(precs) / len(precs))

	print("Ranking MRR")	
	for score_name in score_names:
		mrrs = mrr_list[score_name]

		print(score_name, sum(mrrs) / len(mrrs))

	print("Auc")
	for score_name in score_names:
		aucs = auc_list[score_name]

		print(score_name, sum(aucs) / len(aucs))

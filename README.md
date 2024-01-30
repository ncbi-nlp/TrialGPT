# TrialGPT: Matching Patients to Clinical Trials with Large Language Models

## Introduction

Clinical trials are often hindered by the challenge of patient recruitment. In this work, we introduce TrialGPT, a first-of-its-kind large language model (LLM) framework to assist patient-to-trial matching. Given a patient note, TrialGPT predicts the patient’s eligibility on a criterion-by-criterion basis and then consolidates these predictions assess the patient’s eligibility for the target trial. We evaluate the trial-level prediction performance of TrialGPT on three publicly available cohorts of 184 patients with over 18,000 trial annotations. We also engaged three physicians to label over 1,000 patient-criterion pairs to assess its criterion-level prediction accuracy. Experimental results show that TrialGPT achieves a criterion-level accuracy of 87.3% with faithful explanations, close to the expert performance (88.7%–90.0%). The aggregated TrialGPT scores are highly correlated with human eligibility judgments, and they outperform the best-competing models by 32.6% to 57.2% in ranking and excluding clinical trials. Furthermore, our user study reveals that TrialGPT can significantly reduce the screening time (by 42.6%) in a real-life clinical trial matching task. These results and analyses have demonstrated promising opportunities for clinical trial matching with LLMs such as TrialGPT.


## Configuration

The code has been tested with Python 3.9.13 using CentOS Linux release 7.9.2009 (Core). Please first install the required packages by (this would take only several minutes on a normal computer):
```bash
pip install -r requirements.txt
```

Please also replace the placeholders with your base URL and API key in Microsoft Azure OpenAI Services in `config.py`:
```
config = {
	"api_type": "azure",
	"api_base": "YOUR_API_BASE_URL",
	"api_version": "2023-07-01-preview",
	"api_key": "YOUR_API_KEY"
}
```

## Datasets

We provide the pre-processed datasets of three publicly available cohorts in `./datasets`, including:
- `./datasets/trial_sigir.json` for the SIGIR cohort
- `./datasets/trial_2021.json` for the TREC Clinical Trials 2021 cohort
- `./datasets/trial_2022.json` for the TREC Clinical Trials 2022 cohort

We also put a pre-processed set of the used clinical trials in `./datasets/trial2info.json`.

## Step 1: Criterion-level Prediction

The first step of TrialGPT is to generate the criterion-level predictions, which include (1) the explanation of patient-criterion relevance, (2) locations of relevant sentences, and (3) the eligibility predictions.

Run the following code to get the GPT-4-based TrialGPT results for the three cohorts (depending on the API response rate, matching a patient and a clinical trial normally takes less than 30s):
```bash
# format: python run_matching.py {split} {model}
python run_matching.py sigir gpt-4
python run_matching.py 2021 gpt-4
python run_matching.py 2022 gpt-4
```

## Step 2: Trial-level Aggregation

The second step of TrialGPT is to aggregate the criterion-level predictions to get trial-level scores, including one score for relevance and one score for eligibility.

Please make sure that the step 1 results are ready before running the step 2 code (depending on the API response rate, aggregating a patient-trial pair normally takes less than 20s):
```bash
# format: python run_aggregation.py {split} {model}
python run_aggregation.py sigir gpt-4
python run_aggregation.py 2021 gpt-4
python run_aggregation.py 2022 gpt-4
```

## Step 3: Computing Performance

The third step is to compute the performance of different linear features, LLM features, and the combined features.

Please make sure that the step 1 and step 2 results are ready before running the step 3 code:
```bash
# first convert the results of each split into a csv file
# format: python convert_results_to_csv.py {split} {model}
python convert_results_to_csv.py sigir gpt-4
python convert_results_to_csv.py 2021 gpt-4
python convert_results_to_csv.py 2022 gpt-4

# then compute the results
# format: python get_ranking_results.py {model}
python get_ranking_results.py gpt-4
```

An example output is:
```bash
Ranking NDCG@10
comb 0.8164884118874282
% inc 0.6332474730345071
% not inc 0.5329210870830088
% exc 0.43696962433262426
% not exc 0.45405418648143114
bool not inc 0.5329768607974994
bool exc 0.43696962433262426
random 0.37846131596973925
eligibility 0.7065496001369167
relevance 0.7338932013178386
Ranking Prec@10
comb 0.7327619047619052
% inc 0.5749776817540412
% not inc 0.4977844888166035
% exc 0.4148365417832818
% not exc 0.4310829292061639
bool not inc 0.4977844888166035
bool exc 0.4148365417832818
random 0.36647619047619046
eligibility 0.5659880952380945
relevance 0.552433886908542
Ranking MRR
comb 0.9098095238095236
% inc 0.3827687074829934
% not inc 0.019997732426303858
% exc 0.0009523809523809524
% not exc 0.020113378684807254
bool not inc 0.019997732426303858
bool exc 0.0009523809523809524
random 0.5900770975056686
eligibility 0.8301904761904761
relevance 0.7437573696145123
Auc
comb 0.774898491501416
% inc 0.6524326107402266
% not inc 0.6561815920536348
% exc 0.6512699942037056
% not exc 0.6279445988475326
bool not inc 0.6559597180944899
bool exc 0.6521852962178314
random 0.49775549502869065
eligibility 0.6377132521512072
relevance 0.6495563326979852
```

## Acknowledgments

This work was supported by the Intramural Research Programs of the National Institutes of Health, National Library of Medicine.

## Disclaimer

This tool shows the results of research conducted in the Computational Biology Branch, NCBI/NLM. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.

## Citation

If you find this repo helpful, please cite GeneGPT by:
```bibtex
@article{jin2023matching,
  title={Matching patients to clinical trials with large language models},
  author={Jin, Qiao and Wang, Zifeng and Floudas, Charalampos S and Sun, Jimeng and Lu, Zhiyong},
  journal={ArXiv},
  year={2023},
  publisher={ArXiv}
}
```

# TrialGPT

## Configuration

Please first replace the placeholders with your base URL and API key in Microsoft Azure OpenAI Services in `config.py`:
```
config = {
	"api_type": "azure",
	"api_base": "YOUR_API_BASE_URL",
	"api_version": "2023-07-01-preview",
	"api_key": "YOUR_API_KEY"
}
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

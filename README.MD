

# INSTALLATION

1. create a virtual environment to install the packages in and run the project:         
```
python3 -m venv .project_env
```

2. install necessary packages in the right versions within virtual env.: 
```      
pip install -r requirements.txt
```

If you get a runtime error, try "pip --timeout=1000 install -r requirements.txt"

3. install spacy en_core_web_trf pipeline (https://spacy.io/usage)
```  
python -m spacy download en_core_web_trf
```  



# EXECUTION 

- create an "input" and "results" folder on the top level of the project as well as "intermediate_results" under src 

- upload your regulatory and realization documents as .txt files in the corresponding input sub-folder (check the file_paths script if uncertain about folder structure); the input should be one .txt per Section of the document with the section title = .txt title and content being the text of the Section. 

- under input --> "defined_word_lists", create mapping lists according to your regulatory document and realization

- for the latest text cleaning approach, we changed to notebooks as coreferee gave errors in OOP. Thus, as a first step you need to run: "NEW_preprocessing_optionA _rea.ipynb" and "NEW_preprocessing_optionA _rea.ipynb"

- select your model and parameter choices in main and run main.py with "step_1 = True" in the parameter choices

- run step 2a in the refined phrase extraction project "https://github.com/CatherineSai/textual_constraint_phrase_extraction"
  STEP 2a) is done in another repository as it delivers better results with another spacy version (which is not compatible with a package needed for preprocessing in this script)
        - As Input for that script use the output of step 1 (saved in INTERMEDIATE_DIRECTORY as "gdpr_rea_preprocessed_optiona.xlsx" and "gdpr_reg_preprocessed_optiona.xlsx"). 
        - Copy the output to the path "STEP_TWO_EXTRACTION_RESULTS" --> "gs_reg_rea_extracted_phrases.xlsx"

- run main.py with "step_2 = True" in the parameter choices (this will run the second part of the script, after the phrase extraction was performed in the other project)

- Note: if you want to run the legal-s-bert model, you have to train it (recommended on google colab) with the notebook "legal_sbert_nli_traning.ipynb" 
        Save the resulting model under INTERMEDIATE_DIRECTORY/models/legal_sbert_nli_model. 
        The code for the training was taken from N. Reimers github https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/nli/training_nli_v2.py and adjusted to our domain (Legal-BERT). In our proposed scientific paper N.Reimers corresponding publications are referred to.    




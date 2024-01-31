# Clinical Data Extraction and Summarization

Access to organized medical information in clinical datasets is limited, preventing effective data exploitation. I have developed an unified end-to-end tool that uses Named Entity Recognition (NER) for entity extraction, document search based on a query, and clinical document summarization using a Large Language Model (LLMs) to allow healthcare professionals to retrieve and comprehend relevant medical data quickly.

## Workflow

The detailed workflow is as follows:

#### Step 1
<p align="center">
<img src="./output/image.png" alt="image" width="500"/>
</p>

* Pre-trained [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) model is fine-tuned on [Maccrobat data](https://figshare.com/articles/dataset/MACCROBAT2018/9764942), a medical entity dataset, for Named-Entity-Recognition (NER) task.
* The fine-tuned *distilbert-ner* model is hosted on HuggingFace model hub. The model can be accessed from [here](https://huggingface.co/SahuH/distilbert-ner)
* The complete training code can be accessed in `NER_training.ipynb` notebook

### Step 2
<p align="center">
<img src="./output/image-1.png" alt="image-1" width="400"/>
</p>

 * Here, given a medical report, the fine-tuned *distilbert-ner* model predicts the medical entities in the report
* *distilbert-ner* model also outputs contextual embeddings for the input medical report, which would be used for relevant information retrieval
* Given a query by the user, [FAISS](https://github.com/facebookresearch/faiss) is used to retrieve most relevant sentences in the medical report
* As a final step, summary of the retrieved sentences is generated using pre-trained T5 (*t5-base*) model
* Step 2 integrates all the tasks of the project. Run `run.ipynb` to execute Step 2


`example_run.pdf` contains a sample run of the project for an example 


## Dependencies
```         
pip install -r requirements.txt
```

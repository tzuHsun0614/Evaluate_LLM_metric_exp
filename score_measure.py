import os, math
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import pipeline
from evaluate import load
from tqdm import tqdm
import math
from argparse import ArgumentParser

MODEL_NAME = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
DATASET_NAME = "squad"
BATCH_SIZE = 64
BOOL_NO_ANSWER = False                 
TASK_TYPE = "question-answering"
METRIC = ""

#以為兩種LLM差不多 我實在是錯得離譜
def gen_chunk(dataset, step):
    for i in range(0, len(dataset), step):
        yield dataset[i:i+step]

def qa_model_measure():
    dataset = load_dataset(DATASET_NAME, split="validation")

    LLM = pipeline(
    TASK_TYPE,
    model=MODEL_NAME,
    device=0,
    handle_impossible_answer=True
    )
    metric = load(METRIC)
    num_batches = math.ceil((len(dataset)) / BATCH_SIZE)

    for chunks in tqdm(gen_chunk(dataset, BATCH_SIZE),total=num_batches, desc="Processing batches"):

        
        outputs = LLM(question = chunks["question"],context = chunks["context"],batch_size=BATCH_SIZE)

        ground_truth = chunks["answers"]
        
        ids = chunks["id"]
        
        
        #Accidiently found that hugging face's api might return a dict in the last one, returning list of dicts in other cases
        # print(f"length of each: {len(ids)}, {len(outputs)}, {len(ground_truth)}")
        if isinstance(outputs, dict):
            outputs = [outputs]
        predictions = []
        references = []

        for id,predict,gt in zip(ids,outputs,ground_truth):
            predictions.append({
                "id": id,
                "prediction_text":predict.get("answer") ,
                **({"no_answer_probability": 0.0} if BOOL_NO_ANSWER else {})
            })
            references.append({
                "id": id,
                "answers": gt,
            })
            metric.add_batch(predictions=predictions, references=references)
    result = metric.compute()
    print(result)  


def summarization_model_measure():
    BATCH_SIZE = 4
    if DATASET_NAME == "cnn_dailymail":
        dataset = load_dataset(DATASET_NAME,"3.0.0" ,split="validation")
    else:
        dataset = load_dataset(DATASET_NAME, split="validation")
    LLM = pipeline(
    TASK_TYPE,
    model=MODEL_NAME,
    device=0,
    )
    LLM.tokenizer.model_max_length = getattr(LLM.model.config, "max_position_embeddings", 1024)
    
    rouge_metric = load("rouge")
    bertscore_metric = load("bertscore")

    num_batches = math.ceil((len(dataset)) / BATCH_SIZE)
    for chunks in tqdm(gen_chunk(dataset, BATCH_SIZE),total=num_batches, desc="Processing batches"):

        if DATASET_NAME == "cnn_dailymail":
            outputs = LLM(chunks["article"],truncation=True,max_new_tokens=70,batch_size=BATCH_SIZE)
            ground_truth = chunks["highlights"]
        else:
            outputs = LLM(chunks["summary"],truncation=True,max_new_tokens=70,batch_size=BATCH_SIZE)
            ground_truth = chunks["summary_text"]
        preds = [out['summary_text'] for out in outputs]
        

        rouge_metric.add_batch(predictions=preds, references=ground_truth)
        bertscore_metric.add_batch(predictions=preds, references=ground_truth)
        
        
        
    rouge_scores = rouge_metric.compute(use_stemmer=True, rouge_types=["rouge1","rouge2","rougeL","rougeLsum"])
    bs = bertscore_metric.compute(predictions=[], references=[], lang="en", rescale_with_baseline=True)  
    bertscore = {
        "precision": sum(bs["precision"])/len(bs["precision"]),
        "recall":    sum(bs["recall"])/len(bs["recall"]),
        "f1":        sum(bs["f1"])/len(bs["f1"]),
    }
    print("ROUGE:", rouge_scores)
    print("BERTScore:", bertscore)


    


def get_argument():
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, help='Two tasks are supported: QA (Question Answering Model) and task (Text Summarization Model)')
    parser.add_argument('--model', type=str)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_argument()
    if (args.task == 'QA'):

        TASK_TYPE = "question-answering"
        if args.model == "BERT":
            print("Using BERT model for Question Answering task")
            MODEL_NAME = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
            DATASET_NAME = "squad"
            BOOL_NO_ANSWER = False
            METRIC = "squad"
        elif args.model == "ROBERTA":
            print("Using ROBERTA model for Question Answering task")
            MODEL_NAME = "deepset/roberta-base-squad2"
            DATASET_NAME = "squad_v2"
            BOOL_NO_ANSWER = True
            METRIC = "squad_v2"
        qa_model_measure()
    elif (args.task == 'SUM'):

        TASK_TYPE = "summarization"

        if args.model == "BART":
            print("Using BART model for Text Summarization task")
            MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
            DATASET_NAME = "cnn_dailymail"
            BOOL_NO_ANSWER = False
        elif args.model == "booksum":
            print("Using booksum model for Text Summarization task")
            MODEL_NAME = "cnicu/t5-small-booksum"
            DATASET_NAME = "kmfoda/booksum"
            BOOL_NO_ANSWER = False
        summarization_model_measure()
    else:
       print("Only two tasks are supported: QA (Question Answering Model) and SUM (Text Summarization Model)")

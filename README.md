# Evaluate_LLM_metric_exp

# Start UP
Uses Docker 
``` docker build -f docker/Dockerfile -t llm-exp . ```


``` docker run llm-exp --task <QA or SUM> --model <model_name> ```


# Args 

--task having two options, QA for Question answering model and SUM for Text Summarization model

--model having two cases
QA model having options BERT and ROBERTA
SUM model having options BART and booksum


Not sure if the result needs two model running on one dataset or each model evaluate it's validation data
so in presentation I'll present models running same dasaset, but in program it'll be evaluate by it's own validation set
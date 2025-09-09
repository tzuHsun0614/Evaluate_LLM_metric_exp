# Evaluate_LLM_metric_exp

# Start UP
Uses Docker 
``` docker build -f docker/Dockerfile -t llm-exp . ```
``` docker run llm-exp --task SUM --model BART ```

# Args 

--task having two options, QA for Question answering model and SUM for Text Summarization model

QA model having options BERT and ROBERTA
SUM model having options BART and booksum


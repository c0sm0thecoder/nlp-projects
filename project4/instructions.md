Sentiment Analysis using BERT, QA Systems with BiDAF and BERT

## Task 1. Sentiment Analysis using BERT (30%). 

Analyze any open source Fine tune Bert model for Sentiment Analysis, for example: https://www.analyticsvidhya.com/blog/2021/12/fine-tune-bert-model-for-sentiment-analysis-in-google-colab/

What are inputs and outputs of this model?
How many classes does it have?
What is the size of input?
Is model case sensitive (if yes how it affects to the accuracy)?
Is it possible to use this model for agglutinative languages (Azerbaijani)?

---
## Task 2: Reading Comprehension System (50%)

Objective: In this part of the project, you will focus on developing a system capable of answering questions based on a given context passage. You will implement and integrate a Bidirectional Attention Flow (BiDAF) model and leverage the contextual understanding capabilities of a pre-trained BERT-Base model to achieve this.

Tasks:

1.BiDAF Implementation: (10%)
Implement the BiDAF architecture using either TensorFlow or PyTorch.
The model should take a question and a context passage as input.
The model should output the start and end positions of the answer within the context passage.
2.BERT-Base Integration: (20%)

Utilize a pre-trained BERT-Base model.
Generate contextualized word embeddings for both the question and the context.
Integrate these embeddings into the BiDAF model.
Analyze how BERT embeddings affect the model's performance compared to traditional word embeddings (like GloVe or Word2Vec).
3.Training and Evaluation: (20)

Train the BiDAF model (with or without BERT embeddings) on a suitable reading comprehension dataset (e.g., SQuAD, CoQA).
Evaluate the model's performance using metrics such as Exact Match (EM) and F1-score.
 
---
## Task 3: Write a report (20%).

Extra Task . Create UI for the program results (20%)
---
Presentations

Each team should prepare a presentation, and be prepared to give a short explanation. For each presentation a time slot of 10 minutes is scheduled (5 minutes for presentation + 5 minutes for Q&A).

Final report

Final project write-ups can be at most 5 pages long (including appendices and figures). We will allow for extra pages containing only references. Please include a section that describes what each team member worked on and contributed to the project. Each team member will be addressed the questions related with project's theoretical model and program code accordingly.

Your project report should include the following information:

Motivation: What problem are you tackling, and what's the setting you're considering?
Method: What machine learning techniques have you tried and why?
Experiments: Describe the experiments that you've run, the outcomes, and any error analysis that you've done. You should have tried at least one baseline. Note: negative results that indicate something did not work are welcome.
Submission

The final version of the project report accompanied by the all source code, slides, relevant data and experimental results should be submitted to the Balckboard System after presentation. 


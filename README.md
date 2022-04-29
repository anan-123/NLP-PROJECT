# NLP-PROJECT

## RUN CODE
 1. statistical N-gram model  
           python3 statistical\ N-gram.py
 2. Baseline readability scores
           pip install textstat
           python3 compute_baseline_scores.py
 3.Run the NLP_Project_LSTM_based_NN.ipynb
           load the weights using 
           checkpoint_path = "./train_ckpt/cp.ckpt"
           model_ckpt2 = create_model()
           model_ckpt2.load_weights(checkpoint_path)
## Measure Text Fluency

We aim to measure the fluency of text in corpus. Fluency is commonly considered as one of the dimensions of text quality of MT. Fluency measures the quality of the generated text (e.g., the target translated sentence) i.e how much a sentence is perceived as natural by a human reader, without taking the source into account. It accounts for criteria such as grammar, spelling, choice of words, and style. To understand text fluency we first need to check how readable the text is. The readability of text depends on its content (the complexity of its vocabulary and syntax). It focuses on the words we choose, and how we put them into sentences and paragraphs for the readers to comprehend.
We will use rule based  5 readability index scores, statistical model scores and neural network model scores over the same corpus for each sentence .Then we will normalize each score.Then get a weighted mean score combining all the scores to get best score.Maximum weight is given to the NLM model.

Weights are assigned like below:
 For  readability index score -> weight of 2 
 For statistical model score -> weight of  3
 For NLM model score -> weight of  5

Fluency score = (2*(R1 + R2 + R3 + R4 + R5) + 3*(ST) + 5*(NLM))/(2*5 + 3 + 5)

Then each weighted mean score is scaled to 10 ; low score means better in fluency and can be perceived better.
Then we randomly selected 100 sentence from corpus;and manually calculated Fluency score from scale of 1 to 10 following below rules:



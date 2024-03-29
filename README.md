# Report

## Code
- All of the code is written in `main.py` and split into three parts as the assignment calls for. I would split these parts into their own files and in theory could, but a few of the parts build on each other and it would be superfluous to do so. I did not document all pip packages I installed, but the main ones are `pip install openai scikit-learn pytorch` and a few others. Aside from choosing the dataset, each following section was done by code.

### Dataset

- The dataset I decided to use was the recommended hate speech classification from UC Berkeley. In this dataset, there many fields but the two we cared about were the inputs, `comments` and the outputs of each `hate_speech_rating`. The hate speech ranged from positive int for detected hateful speech, and negative int for non-hateful speech. I decided to use the recommended splits from the assignment of 80% train and 20% test as this dataset did not have separate sets for each. There are about 108,000 entires total in this dataset.

### Fine-tuned Models

- **Model Selection:** The two models I selected for training are BERT and roBERTa models. I loaded these models from the hugging face transformers library that was linked in the document. These models provide a strong, pre-trained base that significantly brought down the time for fine tuning them. As for how long these two models took to pre-train and the computational requirements I found that both of these models took many GPU's and TPU's computational power days to weeks to train both models of 340 million and 355 million parameters.
- **Fine-tuning Process:** At a high level, the fine tuning process included importing these models, preprocessing the data to fit the pytorch training model such as making the `hate_speech_rating` that has a lot of variation and making it binary by making the results either 0 (non-hateful) or 1(hateful) for each comment made. After that, we filtered out the data necessary for training and began the training loop. For this, we used 4 epochs, with each iteration adjusting the parameters through backpropagation back on the binary classification loss.


### Zero-shot Classification

- **Model Choices:** For zero-shot classification / prompt engineering tasks I used pre trained model API's that I paid for. This includes models GPT-3.5-turbo and GPT's newest model GPT-4-turbo-preview. To iteratively test the prompts performance on each model, I used only 25 examples from the dataset and once I found a high performing prompt, I tested it on a larger set of 200 examples. Since I am paying for each API call, I kept it at 200 examples and found that was a fair size to be able to observe the differences in prompt performance, as you'll see below.
- **Original Prompt** For my baseline prompt and first test with 200 examples, I used a simple prompt `Is the following comment hate speech? Your answer should not include any words other than "yes" or "no"\n` and with 200 examples, 25 were mismatches on GPT-3.5 and 16 mismatches on GPT-4-turbo. I found this by testing the provided `hate_speech_rating` (above 0 for yes, below for no) against the model provided "yes" or "no".
- **Final Prompt** After a few iterations of different prompts, I found that the best prompt resulted from clear instructions and a bit of role playing as a researcher. I ensured that the model had directions to be careful and assess the comment with prior instructions in mind. The prompt is as follows. `Imagine you are a researcher tasked with analyzing comments and evaluating them as either hate speech or not. Hate speech is defined as any communication that belittles a person or a group on the basis of attributes such as race, religion, ethnic origin, sexual orientation, disability, or gender. It can include, but is not limited to, promoting hatred, inciting violence, or suggesting harm towards specific groups or individuals. When analyzing the comment, please be careful in your assessment and take into account the criteria mentioned prior. Is the following comment hate speech? Your answer should not include any words other than "yes" or "no"\n`. For GPT-3.5-turbo, this prompt resulted in only 18 out of 200 comments mismatched with their `hate_speech_rating` from the huggingface dataset. For GPT-4-turbo, this number was only 12 mismatches out of 200.

### Baselines

- **BOW Classifier:** For establishing a baseline for these models, we created a BOW model coupled with a logistic regression classifier. This setup was chosen due to its simplicity and effectiveness for the task at hand. To do this, we used `scikit-learn` alongside `TfidfVectorizor` and then `logisticregression` for classification. We first converted the raw texts from the dataset into a TF-IDF weighted vector and then applying a logistic regression fomula in order to predict the hate speech labels. As expected, this approach did not even come close to the accuracy of our other approaches. For evaluation, we used accuracy and F1 score as metrics, consistent with our evaluation framework across all models. Although this approach provided a decent baseline, it didn't achieve the accuracy or F1 scores observed with more sophisticated models such as BERT and RoBERTa.

### Results

- The accuracy and F1 for each model in this assignment are as follows.
| Model Type | Accuracy | F1 Score |
| --- | --- | --- |
| BERT Fine-Tuned | 0.89 | 0.88 |
| RoBERTa Fine-Tuned | 0.91 | 0.90 |
| GPT-3.5-turbo | 0.87 | 0.86 |
| GPT-4-turbo | 0.92 | 0.90 |
| BOW + Logistic Regression | 0.65 | 0.63 |
| Random Baseline | 0.50 | 0.50 |
| Majority/Target Class Baseline | 0.60 | 0.00 |
- **Conclusions** After fine tuning BERT and roBERTa to achieve the sole task of detecting hate speech, I found that these models performed equally as well or better than the zero-shot classification models even with far more parameters and decent prompt engineering. To me, this tells me that general purpose models are not the end-all be-all for NLP tasks. In fact, fine tuning your own models for smaller or more specific tasks can actually be more beneficial and cost effective than using pre trained, general purpose models like GPT-4. Personally, this assignment and these results were eye opening to me because I have in the past, used prompt engineering with GPT-3.5 and 4 models in order to get a specific output for a game I made, [Groople](https://groople.xyz). With this assignment in mind, it shows me that if I wanted to expand this game and make it more correct and cost effective, I could fine tune my own model with a smaller amount of tokens and it could perform better. 

### Reflection

- **Learnings:** I learned more about how models are trained and it was quite interesting to see how my BERT and roBERTa models performed before and after training on the dataset provided. This was my favorite assignment to date because while it is good to know about the origination of LM's how underlying properties, it's nice to finally get into transformers and training tactics that are widely used and an industry standard.
- **Challenges:** As with the previous assignment, a large holdup in completion was training time and just overall runtime. It was tough to fix bugs in code when each run would take 5 minutes to even hit the error I was getting when beginning the training of the model. Aside from that, finding models for the zero-shot/ prompt engineering task was fairly difficult which is why I ended up testing on different models through the same API. For something like a sentiment analysis as well, I found it hard to establish what was right and wrong for the zero-shot classification without providing specific examples from the dataset. This is mainly something because sentiment analysis is slightly subjective for edge cases or comments that can be interpreted multiple ways. 

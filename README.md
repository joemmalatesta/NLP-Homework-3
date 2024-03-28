file:///D:/school/Winter%2024/NLP/NLP_Homework_3_W24.pdf

# Deliverables

This section outlines the expected deliverables for the NLP Homework 03, focusing on the use of pre-trained transformer-based models for hate speech detection. The assignment entails fine-tuning BERT and RoBERTa models, comparing them with zero-shot approaches and a BOW classification method, and summarizing the insights gained.

## Code

- The submitted code will be primarily in Python, utilizing libraries such as Huggingface for model fine-tuning and scikit-learn for the BOW classifier.
- Documentation will accompany the code, providing clear instructions on setup, requirements, and execution steps. A README file will detail the structure of the submission, outlining the purpose of each script and how to replicate the experiments.
- The code will be made available through a GitHub repository, ensuring ease of access and review.

## Report

The report will provide a comprehensive overview of the process, findings, and learnings from the assignment, structured as follows:

### 1. Dataset

- **Description:** The section will detail the hate speech detection task, the source and structure of the dataset, and why it presents a meaningful challenge for text classification.
- **Evaluation Metrics:** Metrics used for evaluation, such as F1 score or accuracy, will be justified.
- **Data Splits:** A table summarizing the distribution of data across training and testing sets, including class imbalances if any.

### 2. Fine-tuned Models

- **Model Selection:** Rationale behind choosing BERT and RoBERTa for fine-tuning.
- **Fine-tuning Process:** Steps taken to adapt these models to the hate speech detection task, including any hyperparameter adjustments.
- **Compute Resources:** A brief overview of the computational resources utilized for training and fine-tuning.

### 3. Zero-shot Classification

- **Model Choices:** Discussion on the selection of models for zero-shot classification, focusing on their inherent capabilities without fine-tuning.
- **Prompt Engineering:** Exploration of the prompts crafted to elicit accurate classifications from the zero-shot models, including failed attempts and the rationale behind the successful prompts.

### 4. Baselines

- **BOW Classifier:** Setup and rationale for the choice of features and classifier type, ensuring reproducibility.
- **Simple Baselines:** Explanation of the random and majority/target-class baselines, providing context for their inclusion and the insights they offer relative to more sophisticated models.

### 5. Results

- **Comparative Analysis:** A table consolidating the performance of all models across the chosen evaluation metrics, followed by a thorough analysis of these results.
- **Insights:** Observations on model performance, including any surprising findings or patterns, and practical recommendations for similar tasks.

### 6. Reflection

- **Learnings:** Reflections on the key takeaways from the assignment, both in terms of technical skills and understanding of NLP model capabilities.
- **Challenges:** A candid discussion of any obstacles encountered during the assignment and the strategies employed to overcome them.

The report aims to not only showcase the results but also the thought process and learnings garnered through the completion of this assignment.
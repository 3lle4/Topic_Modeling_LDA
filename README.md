# Topic Modeling: Hogwarts Legacy Reviews
This Git repository contains the code for a topic modeling analysis of reviews for the game "Hogwarts Legacy" in R.
The project is divided into several steps:

# 1. Load data
The data is loaded from the CSV file "hogwarts_legacy_reviews.csv" from Kaggle. Irrelevant columns are removed, and the review text is selected for further analysis.
Here you can find the dataset: https://www.kaggle.com/datasets/georgescutelnicu/hogwarts-legacy-reviews

# 2. Preprocessing
The review text undergoes several cleaning steps, including the removal of emojis, numbers, punctuation, and stop words. The remaining words are tokenized and grouped by document ID.

# 3. Model Building
A Latent Dirichlet Allocation (LDA) model is constructed to identify topics within the reviews. The number of topics is initially set to 20 and later optimised using coherence scores and other metrics.

# 4. Model Optimisation
Two metrics, CaoJuan2009 and Deveaud2014, are used to optimise the number of topics in the LDA model. The final model is chosen with 8 topics based on these metrics.

# 5. Visualisation
Various visualisations are provided, including the distribution of words across topics, the distribution of topics across documents (theta), and the word-topic probability matrix (phi). Additional visualisations like word clouds and bar plots can be explored. [..still working on this part :) ]

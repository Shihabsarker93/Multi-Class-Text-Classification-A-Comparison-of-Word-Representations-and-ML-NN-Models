# Final Project Report – NLP (CSE440)

## 📌 Project Overview
This project explores different **Machine Learning (ML)** and **Neural Network (NN)** models using multiple text vectorization techniques to classify QA text into categories.  
We conducted **22 experiments** with vectorization methods like **TF-IDF, Bag of Words (BoW), GloVe, and Skip-gram** and applied both traditional ML models and deep learning models to evaluate performance.

---

**Course:** CSE440 (Natural Language Processing)  


---

## 📊 Dataset
- **Train Dataset**: 280,000 rows, 2 columns (`QA Text`, `Class`)  
- **Test Dataset**: 280,000 rows, 2 columns (`QA Text`, `Class`)  
- **Target Feature**: `Class` (10 categories, evenly distributed)  
- **Input Feature**: `QA Text` (combined text of *Question Title*, *Question Content*, and *Best Answer*)  

Key statistics:
- Average **title length**: 10.7 words  
- Average **answer length**: 59.2 words  
- **Content missing** in ~45% of cases  

---

## ⚙️ Methodology
1. **Exploratory Data Analysis (EDA)**  
   - Analyzed class distributions, missing values, word frequency patterns.  
   - Concatenated `Question Title + Question Content + Best Answer` → single text string.  

2. **Preprocessing**  
   - Lowercasing, punctuation removal, stopword removal.  
   - Handled missing values with empty strings.  
   - Applied **lemmatization** for normalization.  

3. **Vectorization Methods**  
   - Bag of Words (BoW)  
   - TF-IDF  
   - GloVe (pre-trained embeddings)  
   - Skip-gram (Word2Vec style embeddings)  

4. **Models Used**  
   - **Machine Learning Models**: Logistic Regression, Naive Bayes, Random Forest  
   - **Neural Networks**:  
     - Deep Neural Network (DNN)  
     - SimpleRNN  
     - GRU  
     - LSTM  
     - Bidirectional SimpleRNN  
     - Bidirectional GRU  
     - Bidirectional LSTM  

---

## 🧪 Experiments
- **BoW + [Logistic Regression, Naive Bayes, Random Forest, DNN]** → 4 experiments  
- **TF-IDF + [Logistic Regression, Naive Bayes, Random Forest, DNN]** → 4 experiments  
- **GloVe + [7 Neural Network models]** → 7 experiments  
- **Skip-gram + [7 Neural Network models]** → 7 experiments  

✅ Total = **22 experiments**  

---

## 📈 Results
| Vectorization | Best Model | Best F1-macro Score |
|---------------|-----------|----------------------|
| **BoW**       | Naive Bayes | **0.6622** |
| **TF-IDF**    | Logistic Regression | **0.6878** |
| **GloVe**     | GRU | ~**0.70** |
| **Skip-gram** | GRU | **0.71** |

🔹 **Winner**: **GRU on Skip-gram embeddings** (F1-macro = 0.71)  
🔻 **Worst Model**: Simple RNN on GloVe (F1-macro = 0.165)  

---

## 📝 Conclusion
- Traditional ML models worked best with **BoW (Naive Bayes)** and **TF-IDF (Logistic Regression)**.  
- Neural Network models significantly outperformed ML models when used with **pre-trained embeddings (GloVe, Skip-gram)**.  
- **GRU** consistently outperformed all other models across embeddings.  
- Main limitations:  
  - Limited GPU resources  
  - Time constraints → used fewer layers & epochs  

Future improvements:  
- Use deeper architectures with more hidden layers.  
- Train for more epochs with optimized GPU resources.  
- Experiment with hybrid embeddings (e.g., GloVe + TF-IDF).  

---

## 📚 References
1. Pennington, Socher, & Manning – [GloVe: Global Vectors for Word Representation (EMNLP 2014)](https://nlp.stanford.edu/projects/glove/)  
2. Mikolov et al. – [Efficient Estimation of Word Representations in Vector Space (ICLR 2013)](https://arxiv.org/abs/1301.3781)  
3. Pedregosa et al. – Scikit-learn: Machine Learning in Python, JMLR 2011  
4. F. Chollet et al. – [Keras](https://keras.io), 2015  
5. Google Brain Team – [TensorFlow](https://www.tensorflow.org)  
6. [NLTK: Natural Language Toolkit](https://www.nltk.org)  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/final-nlp-project.git
   cd final-nlp-project


DATASET LINKS:
https://drive.google.com/drive/u/0/folders/1Zef3MdDtm2esIrZQR1eIEgnTPBiVrU6V 

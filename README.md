# MuseumLangID: Automatic Language Identification for Museum Descriptions
Project developed during the Master in AI Developer at Profession AI

## Introduction
MuseumLangID is a Machine Learning project built on Natural Language Processing (NLP) techniques, designed to solve the multilingual management challenges faced by an international museum.  

The goal is to develop an **automatic and scalable model** to rapidly identify the language of descriptive texts for artworks and artifacts. This eliminates the need for manual identification, which is time-consuming and prone to errors.  

This model is a key solution for the museum to expand its collections while maintaining **operational efficiency** and **information accuracy**.

---

## Project Objectives
- **Automatic Identification**: Develop a model capable of automatically classifying the language of a given text.  
- **Multilingual Support**: Ensure support for at least 3 main languages (verified in the dataset, primarily English `en`, French `fr`, and German `de`).  
- **Easy Integration**: Create a solution that is simple to integrate into the museum's existing information systems.  

---

## Technology Stack and Methodology

### Technologies Used
- **Core Libraries**: `pandas`, `numpy`, `nltk`, `re`  
- **Machine Learning**: `scikit-learn` (for dataset splitting, vectorization, and models)  
- **Visualization**: `matplotlib`, `seaborn`  

### Models and Methodology
- **Data Preprocessing**:  
  - Text cleaning (removal of special characters/numbers, conversion to lowercase)  
  - Tokenization using NLTK  

- **Vectorization**:  
  - TF-IDF (Term Frequency-Inverse Document Frequency) to convert texts into numerical representations that capture the importance of words  

- **Classification Models**:  
  - **Multinomial Naive Bayes (MNB)**: A probabilistic classification model, highly effective in text classification contexts  
  - **Random Forest (RF)**: An ensemble algorithm that constructs multiple decision trees  

---

## Results and Performance Metrics

The comparative analysis between the two models showed excellent performance for the Language Identification task:

| Metric                  | Multinomial Naive Bayes (MNB) | Random Forest (RF) |
|--------------------------|-------------------------------|--------------------|
| **Accuracy**             | 0.95 (95%)                   | 0.88 (88%)         |
| **Precision (Weighted)** | 0.95                         | 0.89               |
| **Recall (Weighted)**    | 0.95                         | 0.88               |
| **F1-Score (Weighted)**  | 0.95                         | 0.88               |

---

## Conclusion
The **Multinomial Naive Bayes (MNB)** model proved to be the most effective for this use case, achieving an accuracy of **95%**.  

The confusion matrices show minimal error dispersion, confirming high precision in classification across the supported languages (primarily `en`, `fr`, `de`).  

**In summary:**  
The developed model offers a **scalable, robust, and highly accurate solution** for the multilingual management of museum descriptions.

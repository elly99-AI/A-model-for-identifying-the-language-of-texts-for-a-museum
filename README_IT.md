# MuseumLangID: Identificazione Automatica della Lingua per Descrizioni Museali

## Introduzione
MuseumLangID è un progetto di Machine Learning basato su tecniche di **Natural Language Processing (NLP)**, progettato per risolvere le sfide di gestione multilingue affrontate da un museo internazionale.  

L’obiettivo è sviluppare un modello **automatico e scalabile** in grado di identificare rapidamente la lingua dei testi descrittivi di opere d’arte e reperti. Questo elimina la necessità di un’identificazione manuale, che risulta dispendiosa in termini di tempo e soggetta a errori.  

Il modello rappresenta una soluzione chiave per consentire al museo di ampliare le proprie collezioni mantenendo **efficienza operativa** e **accuratezza delle informazioni**.

---

## Obiettivi del Progetto
- **Identificazione Automatica**: Sviluppare un modello capace di classificare automaticamente la lingua di un testo.  
- **Supporto Multilingue**: Garantire il supporto ad almeno 3 lingue principali (verificate nel dataset: Inglese `en`, Francese `fr`, e Tedesco `de`).  
- **Facile Integrazione**: Creare una soluzione semplice da integrare nei sistemi informativi già esistenti del museo.  

---

## Stack Tecnologico e Metodologia

### Tecnologie Utilizzate
- **Librerie Core**: `pandas`, `numpy`, `nltk`, `re`  
- **Machine Learning**: `scikit-learn` (per suddivisione dataset, vettorizzazione e modelli)  
- **Visualizzazione**: `matplotlib`, `seaborn`  

### Modelli e Metodologia
- **Preprocessing dei Dati**:  
  - Pulizia del testo (rimozione di caratteri speciali/numeri, conversione in minuscolo)  
  - Tokenizzazione con NLTK  

- **Vettorizzazione**:  
  - Utilizzo di **TF-IDF (Term Frequency-Inverse Document Frequency)** per convertire i testi in rappresentazioni numeriche che catturano l’importanza delle parole  

- **Modelli di Classificazione**:  
  - **Multinomial Naive Bayes (MNB)**: Modello probabilistico, molto efficace nei contesti di classificazione testuale  
  - **Random Forest (RF)**: Algoritmo ensemble che costruisce molteplici alberi decisionali  

---

## Risultati e Metriche di Performance

L’analisi comparativa tra i due modelli ha mostrato ottime performance per il compito di identificazione della lingua:

| Metrica                 | Multinomial Naive Bayes (MNB) | Random Forest (RF) |
|--------------------------|-------------------------------|--------------------|
| **Accuratezza**          | 0.95 (95%)                   | 0.88 (88%)         |
| **Precisione (Weighted)**| 0.95                         | 0.89               |
| **Recall (Weighted)**    | 0.95                         | 0.88               |
| **F1-Score (Weighted)**  | 0.95                         | 0.88               |

---

## Conclusione
Il modello **Multinomial Naive Bayes (MNB)** si è dimostrato il più efficace per questo caso d’uso, raggiungendo un’accuratezza del **95%**.  

Le matrici di confusione mostrano una minima dispersione degli errori, confermando un’elevata precisione nella classificazione delle lingue supportate (`en`, `fr`, `de`).  

**In sintesi:**  
Il modello sviluppato offre una soluzione **scalabile, robusta e altamente accurata** per la gestione multilingue delle descrizioni museali.

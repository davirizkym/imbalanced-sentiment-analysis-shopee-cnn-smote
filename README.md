# CNN–SMOTE Based Sentiment Analysis on Shopee Reviews from Google Play Store

This repository contains the implementation of a **sentiment analysis system for Shopee application reviews collected from Google Play Store** using a **Convolutional Neural Network (CNN)** combined with **Synthetic Minority Over-sampling Technique (SMOTE)** to address the **class imbalance problem**.

The primary objective of this study is to evaluate the impact of SMOTE on improving the detection performance of **negative sentiment**, which represents the minority class but is critical for service quality evaluation.

---

## 🔍 Research Contributions
- Real-time scraping of Shopee user reviews from Google Play Store
- Automated sentiment labeling based on user ratings
- Text preprocessing for Indonesian-language reviews
- CNN-based sentiment classification
- Handling imbalanced data using SMOTE
- Comparative evaluation:
  - CNN without SMOTE
  - CNN with SMOTE
- Comprehensive visualization and error analysis

---

## 🧠 Model Architecture
The proposed CNN architecture consists of:
1. **Embedding Layer**
2. **1D Convolutional Layer**
3. **Global Max Pooling**
4. **Fully Connected Layer**
5. **Dropout Layer**
6. **Sigmoid Output Layer**

The model is trained using **Binary Cross-Entropy Loss** and optimized with **Adam optimizer**.  
Early stopping is applied to prevent overfitting.

---

## 🔄 Experimental Workflow
1. Review scraping using `google-play-scraper`
2. Sentiment labeling:
   - Ratings 4–5 → Positive (1)
   - Ratings 1–2 → Negative (0)
   - Rating 3 → Removed
3. Text preprocessing and tokenization
4. Train-test split
5. Model training:
   - CNN without SMOTE
   - CNN with SMOTE
6. Performance evaluation and comparison
7. Visualization and qualitative analysis

---

## 📊 Key Findings
- CNN without SMOTE achieves high overall accuracy but suffers from **poor recall for the negative class**
- CNN with SMOTE:
  - Slightly lower accuracy
  - **Significantly higher recall for negative sentiment**
- This confirms the **accuracy paradox** commonly observed in imbalanced classification problems

---

## 📈 Evaluation Metrics
- Accuracy
- Recall (Negative Class)
- F1-score (Negative Class)
- Confusion Matrix

---

## 📁 Project Structure

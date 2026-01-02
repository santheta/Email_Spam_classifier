# Email Spam Classifier 📧


### 1. About
This project is an Email/SMS Spam Classifier that uses Machine Learning to classify a given message as **Spam** or **Not Spam**.  
It uses the **Multinomial Naive Bayes** algorithm along with **CountVectorizer** for text feature extraction.  
The trained model is deployed as a web application using **Streamlit**, where users can input a message and get real-time predictions.

---

### 2. Why this project?
Spam messages are a real-world problem affecting digital communication.  
This project helped me understand **text preprocessing**, **feature extraction**, **supervised learning**, and **deployment of a machine learning model** as a web application.

---

### 3. Dataset in use
I used a publicly available **SMS Spam dataset** which contains labeled messages as `spam` or `ham`.  
The dataset mainly consists of two columns:
- Message text  
- Label (spam or ham)

I preprocessed the dataset by removing unnecessary columns and converting labels into numeric form.

---

### 4. Why did I choose Naive Bayes?
Naive Bayes works very well for text classification problems because it handles **high-dimensional data efficiently** and performs well even with relatively small datasets.  
It is fast, simple, and suitable for spam detection.

---

### 5. Which Naive Bayes and why?
I used **Multinomial Naive Bayes** because it is designed for **discrete data like word counts**, which is exactly what CountVectorizer produces.

---

### 6. What is CountVectorizer?
CountVectorizer converts text into numerical form by counting the frequency of words in each message.  
These word counts are then used as features for training the machine learning model.

---

### 7. Explain the flow of your project
1. Load and clean the dataset  
2. Convert text data into numeric features using CountVectorizer  
3. Train the Multinomial Naive Bayes model  
4. Save the trained model and vectorizer using pickle  
5. Load them into a Streamlit web application  
6. Predict whether the user-entered text is Spam or Not Spam  

---

### 8. Why did I save the model using pickle?
Pickle allows the trained model to be **saved and reused**, so retraining is not required every time the application runs.  
This improves efficiency and makes deployment possible.

---

### 9. What is Streamlit and why did I use it?
Streamlit is a Python framework used to build **data science and machine learning web applications** easily.  
I used it because it allows fast deployment without needing separate frontend technologies like HTML, CSS, or JavaScript.

---

### 10. Is your project deployed?
The project runs locally using Streamlit and is **deployment-ready**.  
(It can also be deployed on Streamlit Cloud for public access.)

---

### 11. What challenges did you face?
The main challenge was ensuring the trained model was **correctly saved and loaded**.  
I also faced issues related to **dataset structure and file paths**, which helped me understand proper project organization and deployment practices.

---

### 12. What accuracy did you get?
The model achieves good accuracy for spam classification.  
Since Naive Bayes is a probabilistic model, the focus is on correct classification rather than perfect accuracy.  
The accuracy is approximately **95–97%**, which is typical for this dataset.

---

### 13. What are the limitations of your project?
- The model depends on the training data  
- It may misclassify messages with new or unseen patterns  
- It does not capture deep semantic meaning since it is based on word frequency  

---

### 14. How can this project be improved?
- Use **TF-IDF** instead of CountVectorizer  
- Apply deep learning models like **LSTM**  
- Add text preprocessing such as lemmatization  
- Display prediction confidence  
- Support multiple languages  

---

### 15. Is this supervised or unsupervised learning?
This is **supervised learning** because the model is trained using labeled data.

---

### 16. Why didn’t you use deep learning?
Traditional machine learning models like Naive Bayes are **efficient, interpretable, and sufficient** for this problem.  
Using deep learning would increase complexity without significant benefit for this dataset.

---


A machine learning web application built using **Python**, **Scikit-learn**, and **Streamlit** to classify emails/SMS as **Spam** or **Not Spam**.

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- Naive Bayes
- Pandas

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py

import pandas as pd
import matplotlib.pyplot as plt
import string
import nltk

nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


df = pd.read_csv("spam.csv", encoding="latin-1")


df = df.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"])


df.rename(columns={"v1":"label","v2":"message"}, inplace=True)


df["label"] = df["label"].map({"ham":0,"spam":1})


plt.figure(figsize=(5,4))
df["label"].value_counts().plot(kind="bar")
plt.xticks([0,1],["Ham","Spam"],rotation=0)
plt.title("Ham vs Spam")
plt.savefig("graph1_bar.png")
plt.close()


plt.figure(figsize=(5,5))
plt.pie(df["label"].value_counts(),
        labels=["Ham","Spam"],
        autopct="%1.1f%%")
plt.title("Spam Percentage")
plt.savefig("graph2_pie.png")
plt.close()


df["message_length"]=df["message"].apply(len)


plt.figure(figsize=(7,4))
plt.hist(df["message_length"],bins=40)
plt.title("Message Length Distribution")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.savefig("graph3_histogram.png")
plt.close()


plt.figure(figsize=(5,4))
df.groupby("label")["message_length"].mean().plot(kind="bar")
plt.xticks([0,1],["Ham","Spam"],rotation=0)
plt.title("Average Message Length")
plt.savefig("graph4_avg_length.png")
plt.close()


stop_words=set(stopwords.words("english"))
ps=PorterStemmer()

def transform_text(text):
    text=text.lower()

    temp=""
    for ch in text:
        if ch not in string.punctuation:
            temp+=ch

    words=temp.split()

    cleaned=[]
    for word in words:
        if word not in stop_words:
            cleaned.append(ps.stem(word))

    return " ".join(cleaned)

df["transformed_text"]=df["message"].apply(transform_text)


cv=CountVectorizer(max_features=20)
X_words=cv.fit_transform(df["transformed_text"])
sum_words=X_words.sum(axis=0)

freq=[(w,sum_words[0,idx]) for w,idx in cv.vocabulary_.items()]
freq=sorted(freq,key=lambda x:x[1],reverse=True)

plt.figure(figsize=(10,4))
plt.bar([i[0] for i in freq],[i[1] for i in freq])
plt.xticks(rotation=45)
plt.title("Top 20 Words")
plt.tight_layout()
plt.savefig("graph5_top_words.png")
plt.close()


tfidf=TfidfVectorizer(max_features=3000)

X=tfidf.fit_transform(df["transformed_text"]).toarray()
y=df["label"]

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42)


model=MultinomialNB()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("="*40)
print("Accuracy:",accuracy_score(y_test,y_pred))
print("="*40)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
plt.title("Confusion Matrix")
plt.savefig("graph6_confusion_matrix.png")
plt.close()

while True:
    msg=input("Enter Message (exit to quit): ")

    if msg.lower()=="exit":
        break

    msg=transform_text(msg)
    vector=tfidf.transform([msg]).toarray()

    pred=model.predict(vector)

    if pred[0]==1:
        print("Spam")
    else:
        print("Ham")

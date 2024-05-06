from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})


# sanity check route
@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify('pong!')

# Sentimental analysis
@app.route("/analyze", methods=["GET"])
def sentimental_analysis():

    # Userinput
    user_input = request.json['text']
    # This is for testing
    #user_input = "Dog is happy"

    # Starting
    df = pd.read_csv("train.csv", encoding="ISO-8859-1", engine="python", on_bad_lines="skip")
    #df.info()

    # Delete the null values if needed
    df.isnull().sum()

    # Clean the data
    df.dropna(inplace=True)
    df.isnull().sum()

    # Preparing data for training
    X = df['text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Creating and training the model
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC())
    ])

    text_clf.fit(X_train, y_train)

    # Evaluating the data
    predictions = text_clf.predict(X_test)
    #print(classification_report(y_test, predictions))

    # Visualizing the data (disabled for now)
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=text_clf.classes_)
    disp.plot()
    #plt.show()

    # Making predictions and printing result
    #userInput = input("Insert your sentence here:\n")
    prediction = text_clf.predict([user_input])
    return jsonify({"result": prediction[0]})
    print(f"Prediction: {prediction[0]}")


if __name__ == '__main__':
    app.run(debug=True)
import pickle
import traceback
from flask import Flask, request, jsonify
import pandas as pd
import random
import string

app = Flask(__name__)

@app.route('/predict/', methods=['POST']) # Your API endpoint URL would consist /predict
def predict():
    if lr:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(lr.predict(query))
            prediction_str = [str(i) for i in prediction]
            return jsonify({'prediction': prediction_str})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr=pickle.load(open('model.pkl','rb')) # Load "model.pkl"
    print ('Model loaded')
    model_columns=pickle.load(open('model_columns.pkl','rb')) # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)
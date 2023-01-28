from flask import Flask, request
from fastai.vision.all import load_learner
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import numpy as np

app = Flask(__name__)

learn_inf = load_learner('./ml/export.pkl')

@app.route('/', methods=['GET'])
def index():
  return "<a href=\"/predict\"><h1>Go to predictor</h1></a>"

@app.route('/predict', methods=['GET','POST'])
def predict():
  if request.method == 'GET':
    return """
      <div>
        <h1>Predictor</h1>
        <p>Upload an image to predict (will guess between a black bear, grizzly bear, or a teddy bear)</p>
        <form action="/predict" method="post" enctype="multipart/form-data">
          <input type="file" name="file">
          <input type="submit" value="Upload">
        </form>
      </div>  
      """
  else:
    file = request.files['file']
    img_bytes = file.read()
    pred,pred_idx,probs = learn_inf.predict(img_bytes)
    return f"""
      <div>
        <h1>Predictor</h1>
        <p>Result: {pred}</p>
        <p>Probability: {probs[pred_idx]:.04f}</p>
        <a href="/predict">Go back</a>
      </div>
      """



if __name__ == '__main__':
  app.run()
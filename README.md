Title: Ad Filter Engine
About:
An AI-powered browser that detects and blocks malicious URLs in real time. Built as my Master's final project at University of Arkansas at Little Rock.
How it works:
Run app.py to open the browser. Click "Block/Allow Ads" to test 10 random URLs — safe ones open normally, malicious ones get blocked automatically.
The app uses a 1D Convolution + Bidirectional LSTM deep learning model trained on 9000+ URLs with 92% accuracy.
Models tested:

Random Forest — 98% accuracy (best ML model)
1D Conv + BiLSTM — 92% accuracy (best DL model, used in app)

How to run:

Install dependencies: pip install -r requirements.txt
Run: python app.py

Tech used:
Python, TensorFlow, Keras, PyQt5, scikit-lea,XGBoost, pandas, NumPy, BeautifulSoup

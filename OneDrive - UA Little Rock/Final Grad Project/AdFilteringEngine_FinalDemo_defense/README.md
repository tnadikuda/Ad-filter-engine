This project is intended for the College semester research purpose.

Objective of the project is Ad-Filtering using ML and DL techniques.

Run app.py, this launches a Google browser in the initial task.
The browser has a button Block/Allow Ads, which will start streaming of the ads.

For the demo purpose, 10 urls/ads are picked randomly which can be either benign or malicious.
1D Convolution with Bidirectional LSTM giving an accuracy of about 91% is used to classify the ads, without extracting features.

This featureless technique of Deeplearning will enhance the performance of the browser in blocking/allowing the ads with minimum latency.

Each url will be opened in a separate tab, but the content will be displayed/blocked based on the model's prediction.

Before running application by executing the command python app.py in the terminal, execute the command - pip install -r requirements.txt
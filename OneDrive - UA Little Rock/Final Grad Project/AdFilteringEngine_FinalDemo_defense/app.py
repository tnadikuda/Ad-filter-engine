import sys
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtWebEngineWidgets import QWebEngineView
import pandas as pd
from string import printable
import tensorflow as tf
from keras.preprocessing import sequence
import warnings


warnings.filterwarnings('ignore')

model = tf.keras.models.load_model('models/1DConvWiithBiLSTM.h5')


class BrowserTab(QWebEngineView):
    def __init__(self, url):
        super().__init__()
        self.load(QUrl(url))

class BrowserWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.central_layout = QVBoxLayout(self.central_widget)

        self.block_allow_button = QPushButton("Block/Allow Ads", self.central_widget)
        self.block_allow_button.clicked.connect(self.handle_block_allow)
        self.central_layout.addWidget(self.block_allow_button)

        self.tabs = QTabWidget(self.central_widget)
        self.central_layout.addWidget(self.tabs)

        # Add initial tab with Google search
        self.add_tab("https://www.google.com")

        # Add Block/Allow Ads button
      

        self.setWindowTitle("Ad Filtering Engine")
        self.setGeometry(100, 100, 800, 600)

    def add_tab(self, url):
        browser_tab = BrowserTab(url)
        self.tabs.addTab(browser_tab, url)

    def handle_block_allow(self):
        # Placeholder function, replace with your ML model prediction logic
        def is_url_safe_DL(url):
            url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable]]
            max_len=75
            X = sequence.pad_sequences(url_int_tokens, maxlen=max_len)
            prediction = model.predict(X,batch_size=1)
            # Assuming binary classification, you can use a threshold to determine the class
            threshold = 0.5
            result = 0 if prediction > threshold else 1 ##0 if malicious, 1 if benign
            return result

        # Demo URLs
        url_data = pd.read_csv('data/projectDataSet.csv') 
        # Selecting 10 urls,spread across the class for demo purpose.
        grouped_urls = url_data.groupby('class')['url']
        final_sample = grouped_urls.sample(10, random_state=65)
        final_sample = final_sample.sample(frac = 1)
        final_sample_df = final_sample.to_frame()
        final_sample_df.loc[len(final_sample_df.index)] = ['https://www.bloomberg.com/news/newsletters/2023-10-17/australia-news-today-house-price-rebound-bullock-speaks-lng-strikes-off?srnd=premium-asia']
        for index, row in final_sample_df.iterrows():
            url = row['url']
            default_schema="https://"
            if not url.startswith("http://") and not url.startswith("https://"):
                url = default_schema + url
            print(url)
            prediction = is_url_safe_DL(url)
            if prediction == 1:
                #'benign':
                self.add_tab(url)
            else:
                # Replace this with your blocking logic
                self.add_tab(url)
                self.tabs.widget(self.tabs.count() - 1).setHtml("<h1>Blocked: Malicious Content</h1>")



def main():
    app = QApplication(sys.argv)
    window = BrowserWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

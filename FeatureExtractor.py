from urllib.parse import urlparse
import requests
import socket
from bs4 import BeautifulSoup

##This class will extract the features from the url, to use them while training the ML model.

class FeatureExtractor(object):
    def __init__(self):
        pass

    # Data cleaning
    def add_missing_schema(url, default_schema="http://"):
        if not url.startswith("http://") and not url.startswith("https://"):
            return default_schema + url
        else:
            return url

    def with_ip(url):
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            max_chunk_length = 63
            domain_parts = [domain[i:i + max_chunk_length] for i in range(0, len(domain), max_chunk_length)]
            ip_address = ""
            for domain_part in domain_parts:
                chunk_ip_address = socket.gethostbyname(domain_part)
                ip_address += chunk_ip_address + "."
            ip_address = ip_address.rstrip(".")
            if ip_address:
                return True
            else:
                return False
        except socket.gaierror:
            return False

    def dot_count(url):
        dot_count = url.count('.')
        return dot_count

    def at_Symbol_count(url):
        return url.count('@')

    def is_google_index(url):
        google_search_url = f"https://www.google.com/search?q=site:{url}"
        response = requests.get(google_search_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            search_results = soup.find_all("a", href=True)
            for result in search_results:
                if url in result["href"]:
                    return 1
        return 0

    def directory_count(url):
        urldir = urlparse(url).path
        return urldir.count('/')

    def embed_domain_count(url):
        urldir = urlparse(url).path
        return urldir.count('//')

    def digits_count(url):
        dig = [x for x in url if x.isdigit()]
        return len(dig)

    def parameter_count(url):
        parameter = url.split('&')
        return len(parameter) - 1

    def fragments_count(url):
        fragment = url.split('#')
        return len(fragment) - 1

    def hostname(url):
        try:
            parsed_url = urlparse(url)
            if parsed_url.hostname:
                return parsed_url.hostname
            else:
                return None
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

    def is_live(url):
        try:
            response = requests.head(url)
            if response.status_code >= 200 and response.status_code < 300:
                return True
            else:
                return False
        except requests.ConnectionError:
            return False

    def url_len(url):
        return len(url)

    def extract_features(self, url):
        features = {}
        url = FeatureExtractor.add_missing_schema(url)
        features['with_ip'] = FeatureExtractor.with_ip(url)
        features['dot_count'] = FeatureExtractor.dot_count(url)
        features['at_Symbol_count'] = FeatureExtractor.at_Symbol_count(url)
        features['is_google_index'] = FeatureExtractor.is_google_index(url)
        features['directory_count'] = FeatureExtractor.directory_count(url)
        features['embed_domain_count'] = FeatureExtractor.embed_domain_count(url)
        features['digits_count'] = FeatureExtractor.digits_count(url)
        features['parameter_count'] = FeatureExtractor.parameter_count(url)
        features['fragments_count'] = FeatureExtractor.fragments_count(url)
        features['hostname'] = FeatureExtractor.hostname(url)
        features['is_live'] = FeatureExtractor.is_live(url)
        features['url_len'] = FeatureExtractor.url_len(url)
        return features

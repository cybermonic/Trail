import xgboost as xgb
import shap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import config
from feature_extraction.columns import URL_COLS, DOMAIN_COLS, IP_COLS
from train_xgb import XGBoostGPUClassifierAPT

# Function to downsample the data (NumPy arrays)
def downsample_data(X, y, num_samples=1000):
    """Randomly downsample X and y to num_samples for NumPy arrays."""
    if X.shape[0] > num_samples:
        sampled_indices = np.random.choice(X.shape[0], num_samples, replace=False)
        return X[sampled_indices], y[sampled_indices]
    return X, y

# Load pre-trained models
URL_MODEL = xgb.Booster()
IP_MODEL = xgb.Booster()
DOMAIN_MODEL = xgb.Booster()

# Get data for URL model
urls = XGBoostGPUClassifierAPT(config=config,
                               model_title=f"",
                               model_definition='',
                               ioc_type='urls')
X_test_urls, y_test_urls = urls.X_test, urls.y_test

# Get data for IP model
ips = XGBoostGPUClassifierAPT(config=config,
                              model_title=f"",
                              model_definition='',
                              ioc_type='ips')
X_test_ips, y_test_ips = ips.X_test, ips.y_test

# Get data for Domain model
domains = XGBoostGPUClassifierAPT(config=config,
                                  model_title=f"",
                                  model_definition='',
                                  ioc_type='domains')
X_test_domains, y_test_domains = domains.X_test, domains.y_test

# Downsample the data (e.g., 1000 samples)
X_test_urls, y_test_urls = downsample_data(X_test_urls, y_test_urls, num_samples=11000)
X_test_ips, y_test_ips = downsample_data(X_test_ips, y_test_ips, num_samples=11000)
X_test_domains, y_test_domains = downsample_data(X_test_domains, y_test_domains, num_samples=11000)

# Load pre-trained models
URL_MODEL.load_model(Path(config.get('ML_DATA')) / 'urls/pure_model/APT_XGBoost_url_full.json')
IP_MODEL.load_model(Path(config.get('ML_DATA')) / 'ips/pure_model/APT_XGBoost_ip_full.json')
DOMAIN_MODEL.load_model(Path(config.get('ML_DATA')) / 'domain/pure_model/APT_XGBoost_domain_full.json')

# Convert XGBoost boosters into XGBClassifier for SHAP
url_model_classifier = xgb.XGBClassifier()
url_model_classifier._Booster = URL_MODEL

ip_model_classifier = xgb.XGBClassifier()
ip_model_classifier._Booster = IP_MODEL

domain_model_classifier = xgb.XGBClassifier()
domain_model_classifier._Booster = DOMAIN_MODEL

# CHOOSE APT
apt_index = 0  # APT28

# Set figure size for larger graph
plt.figure(figsize=(16, 12))

# Compute SHAP values for URL model with GPU acceleration
url_explainer = shap.TreeExplainer(url_model_classifier, feature_perturbation='tree_path_dependent', model_output='raw', feature_names=URL_COLS)
url_shap_values = url_explainer(X_test_urls)

# Plot SHAP beeswarm for URL model with specified parameters
shap.plots.beeswarm(url_shap_values[:, :, apt_index], max_display=15, show=False, color_bar=True, s=20)
# Adjust the margins to give more space to the left for feature names
plt.subplots_adjust(left=0.4)  # Increase the left margin to 30% of the figure width
plt.title('SHAP Beeswarm Plot for APT28 - URL Model')  # Add title
plt.show()

# Set figure size for larger graph
plt.figure(figsize=(16, 12))

# Compute SHAP values for IP model with GPU acceleration
ip_explainer = shap.TreeExplainer(ip_model_classifier, feature_perturbation='tree_path_dependent', model_output='raw', feature_names=IP_COLS)
ip_shap_values = ip_explainer(X_test_ips)

# Plot SHAP beeswarm for IP model
shap.plots.beeswarm(ip_shap_values[:, :, apt_index], max_display=15, show=False, color_bar=True, s=20)
# Adjust the margins to give more space to the left for feature names
plt.subplots_adjust(left=0.4)  # Increase the left margin to 30% of the figure width
plt.title('SHAP Beeswarm Plot for APT28 - IP Model')  # Add title
plt.show()

# Set figure size for larger graph
plt.figure(figsize=(18, 14))

# Compute SHAP values for Domain model with GPU acceleration
domain_explainer = shap.TreeExplainer(domain_model_classifier, feature_perturbation='tree_path_dependent', model_output='raw', feature_names=DOMAIN_COLS)
domain_shap_values = domain_explainer(X_test_domains)

# Plot SHAP beeswarm for Domain model
shap.plots.beeswarm(domain_shap_values[:, :, apt_index], max_display=15, show=False, color_bar=True, s=20)
# Adjust the margins to give more space to the left for feature names
plt.subplots_adjust(left=0.4)  # Increase the left margin to 30% of the figure width
plt.title('SHAP Beeswarm Plot for APT28 - Domain Model')  # Add title
plt.show()

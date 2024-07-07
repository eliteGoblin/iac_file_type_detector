import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from joblib import dump, load
import scipy

# Function to load files from a directory and label them
def load_files_from_directory(directory, label):
    files_data = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            with open(file_path, 'r') as file:
                try:
                    content = file.read()
                    files_data.append((content, label))
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue
    return files_data

# Function to add manual features
def add_manual_features(text):
    features = {
        'Resources': int('Resources' in text),
        'AWSTemplateFormatVersion': int('AWSTemplateFormatVersion' in text),
        'AWS': int('AWS' in text),
        'Outputs': int('Outputs' in text),
        'Conditions': int('Conditions' in text),
        'Parameters': int('Parameters' in text),
        'Mappings': int('Mappings' in text),
        'Description': int('Description' in text),
        'Metadata': int('Metadata' in text),
        'Microsoft': int('Microsoft' in text),
        'schema': int('schema' in text),
        '$schema': int('$schema' in text),
        'contentVersion': int('contentVersion' in text),
        'resources': int('resources' in text),
        'variables': int('variables' in text),
        'outputs': int('outputs' in text),
        'tasks': int('tasks' in text),
        'hosts': int('hosts' in text),
        'roles': int('roles' in text),
        'vars': int('vars' in text),
        'handlers': int('handlers' in text),
        'environment': int('environment' in text),
        'jobs': int('jobs' in text),  # Common in GitHub Actions
        'services': int('services' in text),  # Common in Docker Compose
        'steps': int('steps' in text),  # Common in GitHub Actions and CircleCI
        'version': int('version' in text)  # Common in various YAML files
    }
    return features

# Check if the model and vectorizer files exist
model_filename = 'ensemble_model.joblib'
vectorizer_filename = 'tfidf_vectorizer.joblib'

if os.path.exists(model_filename) and os.path.exists(vectorizer_filename):
    # Load the saved model and vectorizer
    ensemble_model = load(model_filename)
    vectorizer = load(vectorizer_filename)
    print(f'Model and vectorizer loaded from {model_filename} and {vectorizer_filename}')
else:
    # Load the data using the updated paths
    cloudformation_files = load_files_from_directory('../scripts/generate_test_data/gitignore/iac_files/cloudformation/', 'cloudformation')
    azure_arm_files = load_files_from_directory('../scripts/generate_test_data/gitignore/iac_files/azure_arm/', 'azure_arm_yml')
    ansible_files = load_files_from_directory('../scripts/generate_test_data/gitignore/iac_files/ansible', 'ansible')
    noniac_files = load_files_from_directory('../scripts/generate_test_data/gitignore/iac_files/noniac', 'noniac')

    # Combine all data into a single DataFrame
    data = cloudformation_files + azure_arm_files + ansible_files + noniac_files
    df = pd.DataFrame(data, columns=['content', 'label'])

    # Add manual features to the DataFrame
    df_features = df['content'].apply(add_manual_features)
    df_manual_features = pd.DataFrame(df_features.tolist())

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=1, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Combine TF-IDF features with manual features
    X_train_combined = scipy.sparse.hstack((X_train_tfidf, df_manual_features.loc[X_train.index]))
    X_test_combined = scipy.sparse.hstack((X_test_tfidf, df_manual_features.loc[X_test.index]))

    # Train an ensemble of models
    clf1 = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
    clf2 = LogisticRegression(max_iter=1000)
    clf3 = SVC(probability=True)

    ensemble_model = VotingClassifier(estimators=[
        ('rf', clf1), ('lr', clf2), ('svc', clf3)], voting='soft')
    ensemble_model.fit(X_train_combined, y_train)

    # Evaluate the model
    y_pred = ensemble_model.predict(X_test_combined)
    print(classification_report(y_test, y_pred))

    # Save the trained model to a file
    dump(ensemble_model, model_filename)
    dump(vectorizer, vectorizer_filename)

    print(f'Model and vectorizer saved to {model_filename} and {vectorizer_filename}')

def predict_type(file_path) -> str:
    with open(file_path, 'r') as file:
        content = file.read()
        content_tfidf = vectorizer.transform([content])
        manual_features = add_manual_features(content)
        manual_features_df = pd.DataFrame([manual_features])
        content_combined = scipy.sparse.hstack((content_tfidf, manual_features_df))
        prediction_proba = ensemble_model.predict_proba(content_combined)
        prediction = ensemble_model.predict(content_combined)
        predicted_label = prediction[0]
        max_proba = max(prediction_proba[0])
        if max_proba < 0.7:  # Threshold for unknown
            return "unknown"
        return predicted_label

# Function to check and return the type or unknown
def check_and_predict(file_path) -> str:
    known_types = ['cloudformation', 'azure_arm_yml', 'ansible']
    predicted_type = predict_type(file_path)
    if predicted_type in known_types:
        return predicted_type
    else:
        return "unknown"

# Example usage
# Iterate files in the test/ folder, check their type using the combined model
for file in os.listdir('./test'):
    file_path = os.path.join('./test', file)
    result = check_and_predict(file_path)
    print(f'{file}: predict type {result}')

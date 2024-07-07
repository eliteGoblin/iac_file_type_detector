import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
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

# Load the data using the updated paths
cloudformation_files = load_files_from_directory('../scripts/generate_test_data/gitignore/iac_files/cloudformation/', 'cloudformation')
azure_arm_files = load_files_from_directory('../scripts/generate_test_data/gitignore/iac_files/azure_arm/', 'azure_arm_yml')
ansible_files = load_files_from_directory('../scripts/generate_test_data/gitignore/iac_files/ansible', 'ansible')

# Combine all data into a single DataFrame
data = cloudformation_files + azure_arm_files + ansible_files
df = pd.DataFrame(data, columns=['content', 'label'])

# Split the data for each type
df_cfn = df[df['label'] == 'cloudformation']
df_azure = df[df['label'] == 'azure_arm_yml']
df_ansible = df[df['label'] == 'ansible']

# Function to add manual features
def add_manual_features(text, label):
    features = {
        'Resources': int('Resources' in text),
        'AWSTemplateFormatVersion': int('AWSTemplateFormatVersion' in text),
        'Outputs': int('Outputs' in text),
        'Conditions': int('Conditions' in text),
        'Parameters': int('Parameters' in text),
        'Mappings': int('Mappings' in text),
        'Description': int('Description' in text),
        'Metadata': int('Metadata' in text),
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
        'environment': int('environment' in text)
    }
    return features

def process_and_train(df, label):
    if df.empty:
        print(f"No data available to train the model for {label}.")
        return None, None

    print(f"Training data statistics for {label}:")
    print(df['label'].value_counts())

    df_features = df.apply(lambda x: add_manual_features(x['content'], x['label']), axis=1)
    df_manual_features = pd.DataFrame(df_features.tolist())

    print(f"Manual features for {label} added. Sample features:")
    print(df_manual_features.head())
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)
    
    print(f"Labels distribution in training data for {label}:")
    print(y_train.value_counts())
    print(f"Labels distribution in test data for {label}:")
    print(y_test.value_counts())

    # Ensure there are at least two classes
    if len(y_train.unique()) < 2 or len(y_test.unique()) < 2:
        print(f"Not enough classes to train the model for {label}.")
        return None, None
    
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
    print(f"Classification report for {label} model:")
    print(classification_report(y_test, y_pred))
    
    return ensemble_model, vectorizer

# Train separate models for each type
model_cfn, vectorizer_cfn = process_and_train(df_cfn, 'cloudformation')
model_azure, vectorizer_azure = process_and_train(df_azure, 'azure_arm_yml')
model_ansible, vectorizer_ansible = process_and_train(df_ansible, 'ansible')

def predict_type(file_path, model, vectorizer, label) -> str:
    if model is None or vectorizer is None:
        return f"Model for {label} was not trained due to insufficient data."
    
    with open(file_path, 'r') as file:
        content = file.read()
        content_tfidf = vectorizer.transform([content])
        manual_features = add_manual_features(content, label)
        manual_features_df = pd.DataFrame([manual_features])
        content_combined = scipy.sparse.hstack((content_tfidf, manual_features_df))
        prediction = model.predict(content_combined)
        return prediction[0]

# Example usage
# Iterate files in the test/ folder, check their type using corresponding models
for file in os.listdir('./test'):
    file_path = os.path.join('./test', file)
    result_cfn = predict_type(file_path, model_cfn, vectorizer_cfn, 'cloudformation')
    result_azure = predict_type(file_path, model_azure, vectorizer_azure, 'azure_arm_yml')
    result_ansible = predict_type(file_path, model_ansible, vectorizer_ansible, 'ansible')
    print(f'{file}: CFN predict type {result_cfn}, Azure ARM predict type {result_azure}, Ansible predict type {result_ansible}')

# iac_file_type_detector

Simple ML to detect IAC file types: if CFN, Ansible, Azure ARM or others(unknown)

```s
cd build_model
source .env/bin/activate
pip install -r requirements.txt


python train_one_model.py
Model and vectorizer loaded from ensemble_model.joblib and tfidf_vectorizer.joblib # load from pre-trained model

gha1.yml: predict type unknown
ansible.yml: predict type ansible
k8s1.yml: predict type unknown
cfn2.yml: predict type cloudformation
custom2.yml: predict type unknown
arm.yml: predict type azure_arm_yml
docker-compose-1.yml: predict type unknown
custom1.yml: predict type unknown
circleci1.yml: predict type unknown
cfn.yml: predict type cloudformation
travisci.yml: predict type unknown
```
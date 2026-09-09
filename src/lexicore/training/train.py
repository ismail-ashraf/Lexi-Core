import mlflow
import yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from lexicore.tokenization.bpe_tokenizer import BPETokenizer,BPEVocabulary

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("LexiCore_Risk_Classification")

folder_path = Path(__file__).resolve().parent
config_file = (
    folder_path / 'config' / 'training_config.yaml'
)

with open(config_file,"r",encoding="utf-8") as f:
    config = yaml.safe_load(f)

#===================data path==================
#==============================================
data_path = Path(config['data']['data_path'])

df = pd.read_csv(data_path)#type:ignore
# print(df.head())#type:ignore
#===================mapping the labels=============
#==================================================
label_map = {
    "low" : 0,
    "medium" : 1,
    "high" : 2
}
df['risk_level'] = df['risk_level'].map(label_map) # type:ignore

# print(df.columns)
# Columns -> 'text' ,'risk_level','reason'
#==================split the data=================
#=================================================
X = df[config['data']['X']]
y = df[config['data']['y']]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

X_train,X_val,y_train,y_val = train_test_split(
    X_train,y_train,test_size=0.125,random_state=42,stratify=y_train
)

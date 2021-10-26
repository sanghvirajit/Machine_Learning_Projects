import requests
import pandas as pd
import json

host = 'churn-serving-env.eba-gyzkzxig.eu-central-1.elasticbeanstalk.com'
url = f'http://{host}/predict_flask'

df = pd.read_csv("customer_test.csv")

customer_number = 23

customer = df.iloc[customer_number-1].to_json()
customer = json.loads(customer)

response = requests.post(url, json=customer).json()
print(response)

if response['Fraud'] == True:
    print("It's a fraud claim")
else:
    print("Claim is legit")
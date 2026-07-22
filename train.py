import pandas as pd
import numpy as np
import utilities
import pickle

# 1. Load data
data = pd.read_csv("heart.csv")
X_full = data.drop(columns='target').copy()
y_full = data['target'].copy()

# 2. Train/Test Split
X_train,X_test,y_train,y_test = utilities.custom_train_test_split(X_full,y_full,test_size=0.2,random_state=1)

# 3. Standardize Features
X_train_scaled,X_test_scaled,mean,std = utilities.standardize(X_train,X_test)

# 4. Train Model
model = utilities.LogisticRegression()
model.fit(X_train_scaled,y_train)

# 5.Evaluate Performance
predictions = model.predict(X_test_scaled)
accuracy = np.mean(predictions==y_test)
print(f"Model Accuracy: {accuracy*100:.2f}%")

model_data = {'weights':model.weights,'bias':model.bias,'mean':mean,'std':std}

# 6. Save Model
with open('model.pkl','wb') as f:
    pickle.dump(model_data,f)

print("Model saved successfully as model.pkl! 🎉")
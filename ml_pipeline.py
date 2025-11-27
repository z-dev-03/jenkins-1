import pickle
import sys
import numpy as np
from sklearn.linear_model import LinearRegression

print("=== ML CI Pipeline ===")

# 1. Create training data (y = 2x)
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])

# 2. Train model
model = LinearRegression()
model.fit(X, y)
print(" Model trained")

# 3. Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print(" Model saved")

# 4. Test prediction (input=4, expect=8)
prediction = model.predict([[4]])[0]
print(f" Prediction: {prediction:.1f} (Expected: 8.0)")

# 5. Validate
if abs(prediction - 8.0) < 0.1:
    print(" VALIDATION PASSED")
    sys.exit(0)
else:
    print(" VALIDATION FAILED")
    sys.exit(1)

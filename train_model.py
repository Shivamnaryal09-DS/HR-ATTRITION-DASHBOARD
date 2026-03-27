from hr_attrition_predictor import HRAttritionPredictor
import zipfile

# Unzip the dataset
with zipfile.ZipFile('hr_dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('hr_dataset')

# Initialize the HRAttritionPredictor
predictor = HRAttritionPredictor()

# Train the model on the dataset
predictor.train('hr_dataset')

# Save the trained model
predictor.save_model('hr_attrition_model.pkl')

# Display feature importance
importance = predictor.get_feature_importance()
print('Feature Importance:', importance)
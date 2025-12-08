#!/usr/bin/env python3
"""
Quick Setup for Vehicle Insurance Fraud Detection
Creates minimal working models and data
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from faker import Faker
import os

def create_sample_data():
    """Generate sample insurance data"""
    fake = Faker()
    np.random.seed(42)
    
    print("Generating sample data...")
    
    # Generate 1000 sample records
    data = []
    for i in range(1000):
        record = {
            'months_as_customer': np.random.randint(1, 600),
            'age': np.random.randint(18, 80),
            'policy_state': np.random.choice(['OH', 'IN', 'IL']),
            'policy_csl': np.random.choice(['100/300', '250/500', '500/1000']),
            'policy_deductible': np.random.choice([500, 1000, 2000]),
            'policy_annual_premium': np.random.randint(500, 3000),
            'education_level': np.random.choice(['High School', 'College', 'Masters']),
            'occupation': np.random.choice(['tech-support', 'sales', 'exec-managerial']),
            'incident_type': np.random.choice(['Single Vehicle Collision', 'Multi-vehicle Collision', 'Vehicle Theft']),
            'collision_type': np.random.choice(['Front Collision', 'Rear Collision', 'Side Collision']),
            'incident_severity': np.random.choice(['Minor Damage', 'Major Damage', 'Total Loss']),
            'authorities_contacted': np.random.choice(['Police', 'Fire', 'Ambulance', 'None']),
            'auto_make': np.random.choice(['Toyota', 'Honda', 'Ford']),
            'auto_year': np.random.randint(1995, 2024),
            'total_claim_amount': np.random.randint(1000, 200000),
        }
        
        # Add engineered features
        record['vehicle_age'] = 2024 - record['auto_year']
        record['claim_ratio'] = record['total_claim_amount'] / record['policy_annual_premium']
        
        # Enhanced fraud scoring with extreme differentiation
        fraud_score = 0
        
        # Extreme claim ratios (most critical factor)
        if record['claim_ratio'] > 100: fraud_score += 10  # 100x premium = almost certain fraud
        elif record['claim_ratio'] > 50: fraud_score += 8   # 50x premium = very high risk
        elif record['claim_ratio'] > 25: fraud_score += 6   # 25x premium = high risk
        elif record['claim_ratio'] > 15: fraud_score += 4   # 15x premium = moderate risk
        elif record['claim_ratio'] > 8: fraud_score += 2    # 8x premium = some risk
        elif record['claim_ratio'] > 4: fraud_score += 1    # 4x premium = low risk
        
        # Authority contact critical for severe incidents
        if record['authorities_contacted'] == 'None':
            if record['incident_severity'] == 'Total Loss': fraud_score += 6
            elif record['incident_severity'] == 'Major Damage': fraud_score += 4
            else: fraud_score += 2
        
        # New customer with large claims (red flag)
        if record['months_as_customer'] <= 3:
            if record['total_claim_amount'] > 75000: fraud_score += 7
            elif record['total_claim_amount'] > 50000: fraud_score += 5
            elif record['total_claim_amount'] > 25000: fraud_score += 3
        elif record['months_as_customer'] <= 12:
            if record['total_claim_amount'] > 100000: fraud_score += 4
            elif record['total_claim_amount'] > 75000: fraud_score += 2
        
        # Vehicle age vs claim amount patterns
        if record['vehicle_age'] > 20 and record['total_claim_amount'] > 50000: fraud_score += 5
        elif record['vehicle_age'] > 15 and record['total_claim_amount'] > 75000: fraud_score += 3
        elif record['vehicle_age'] < 3 and record['incident_severity'] == 'Total Loss': fraud_score += 4
        
        # Young drivers with expensive claims
        if record['age'] < 25:
            if record['total_claim_amount'] > 100000: fraud_score += 4
            elif record['total_claim_amount'] > 50000: fraud_score += 2
        
        # Education vs occupation vs claim patterns
        if record['education_level'] == 'High School' and record['total_claim_amount'] > 100000: fraud_score += 2
        if record['occupation'] == 'sales' and record['total_claim_amount'] > 75000: fraud_score += 1
        
        # Create highly differentiated fraud probabilities
        if fraud_score >= 15:
            fraud_probability = 0.92 + np.random.random() * 0.07  # 92-99% - Almost certain fraud
        elif fraud_score >= 12:
            fraud_probability = 0.80 + np.random.random() * 0.10  # 80-90% - Very high fraud
        elif fraud_score >= 9:
            fraud_probability = 0.65 + np.random.random() * 0.12  # 65-77% - High fraud
        elif fraud_score >= 6:
            fraud_probability = 0.45 + np.random.random() * 0.15  # 45-60% - Moderate fraud
        elif fraud_score >= 3:
            fraud_probability = 0.20 + np.random.random() * 0.20  # 20-40% - Low-moderate fraud
        elif fraud_score >= 1:
            fraud_probability = 0.05 + np.random.random() * 0.15  # 5-20% - Low fraud
        else:
            fraud_probability = np.random.random() * 0.08  # 0-8% - Very low fraud
        
        record['fraud_reported'] = 1 if np.random.random() < fraud_probability else 0
        
        data.append(record)
    
    return pd.DataFrame(data)

def train_models(df):
    """Train and save models"""
    print("Training models...")
    
    # Prepare features
    categorical_cols = ['policy_state', 'policy_csl', 'education_level', 'occupation', 
                       'incident_type', 'collision_type', 'incident_severity', 'authorities_contacted', 'auto_make']
    
    # Encode categorical variables
    encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    # Features for training
    feature_cols = ['months_as_customer', 'age', 'policy_deductible', 'policy_annual_premium',
                   'auto_year', 'total_claim_amount', 'vehicle_age', 'claim_ratio'] + categorical_cols
    
    X = df_encoded[feature_cols]
    y = df_encoded['fraud_reported']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save models
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/features.pkl')
    
    print("Models saved successfully!")
    
    # Test accuracy
    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    print(f"Model accuracy: {accuracy:.2%}")
    
    return model, encoders, scaler, feature_cols

def main():
    """Main setup function"""
    print("Vehicle Insurance Fraud Detection - Quick Setup")
    print("=" * 60)
    
    # Generate data
    df = create_sample_data()
    print(f"Generated {len(df)} records")
    
    # Save data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/insurance_claims.csv', index=False)
    print("Data saved to data/insurance_claims.csv")
    
    # Train models
    model, encoders, scaler, features = train_models(df)
    
    print("\nSetup completed successfully!")
    print("\nTo run the application:")
    print("streamlit run app/clean_app.py")
    
    return True

if __name__ == "__main__":
    main()
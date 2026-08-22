import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

# Generate numeric columns
weekly_study_hours = np.random.exponential(scale=8, size=n)
weekly_study_hours = np.clip(weekly_study_hours, 0, 60)
weekly_study_hours = np.round(weekly_study_hours, 1)

attendance_pct = np.random.beta(a=2, b=1.5, size=n) * 100
attendance_pct = np.clip(attendance_pct, 20, 100)
attendance_pct = np.round(attendance_pct, 1)

prev_exam_score = np.random.normal(loc=65, scale=15, size=n)
prev_exam_score = np.clip(prev_exam_score, 10, 100)
prev_exam_score = np.round(prev_exam_score, 1)

mock_test_3 = np.random.normal(loc=55, scale=20, size=n)
mock_test_3 = np.clip(mock_test_3, 0, 100)
mock_test_3 = np.round(mock_test_3, 1)

# Generate categorical columns
income_bracket = np.random.choice(['Low', 'Medium', 'High'], size=n, p=[0.3, 0.5, 0.2])
feedback_options = [
    'Great course, learned a lot',
    'Could use more practice problems',
    'Instructor was very helpful',
    'Pace was too fast',
    'Materials were comprehensive',
    'Need more visual aids',
    'Excellent teaching style',
    'Would recommend to others',
]
feedback_text = np.random.choice(feedback_options, size=n)
# Add ~10% missing feedback using object array to preserve NaN
feedback_text = feedback_text.astype(object)
feedback_missing = np.random.choice(n, size=int(n * 0.1), replace=False)
feedback_text[feedback_missing] = np.nan

# Additional context columns
cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune']
courses = ['Math', 'Physics', 'Chemistry', 'Biology', 'CS']
batch_types = ['Morning', 'Evening', 'Weekend']

city = np.random.choice(cities, size=n)
course = np.random.choice(courses, size=n)
batch_type = np.random.choice(batch_types, size=n)

df = pd.DataFrame({
    'city': city,
    'course': course,
    'batch_type': batch_type,
    'weekly_study_hours': weekly_study_hours,
    'attendance_pct': attendance_pct,
    'prev_exam_score': prev_exam_score,
    'mock_test_3': mock_test_3,
    'income_bracket': income_bracket,
    'feedback_text': feedback_text
})

# Introduce missing values with specific mechanisms

# MCAR: Random missingness in weekly_study_hours (~5%)
mcar_indices = np.random.choice(n, size=50, replace=False)
df.loc[mcar_indices, 'weekly_study_hours'] = np.nan

# MAR: Missing prev_exam_score when attendance_pct < 40
mar_mask = df['attendance_pct'] < 40
df.loc[mar_mask, 'prev_exam_score'] = np.nan

# MNAR: Missing mock_test_3 when mock_test_3 < 30
mnar_mask = df['mock_test_3'] < 30
df.loc[mnar_mask, 'mock_test_3'] = np.nan

# Categorical missingness
cat_missing_indices = np.random.choice(n, size=80, replace=False)
df.loc[cat_missing_indices, 'income_bracket'] = np.nan

# feedback_text missingness was set above (lines 37-40)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('data/raw/student_performance_raw.csv', index=False)

print(f"Dataset created: {len(df)} rows, {len(df.columns)} columns")
print(f"\nMissing values:\n{df.isnull().sum()}")

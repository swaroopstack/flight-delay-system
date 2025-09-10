
pip install pandas scikit-learn matplotlib

import pandas as pd
import random
from datetime import datetime, timedelta


random.seed(42)

# Function to generate random weather conditions
def generate_weather():
    temperature = random.randint(-10, 40)  # Temperature in Celsius
    visibility = random.choice([5, 10, 15, 20, 25])  # Visibility in kilometers
    wind_speed = random.randint(0, 30)  # Wind speed in km/h
    return temperature, visibility, wind_speed

# Function to generate flight status (delayed or on-time)
def generate_flight_status():
    return random.choice(["On-time", "Delayed"])

# Generate sample data for 100 flights
num_flights = 100
flights = []

for i in range(num_flights):
    flight_number = f"FL{random.randint(100, 999)}"
    airline = random.choice(["Delta", "United", "American", "Southwest", "JetBlue"])

    # Generate random departure and arrival times
    departure_time = datetime.now() + timedelta(days=random.randint(0, 30), hours=random.randint(0, 12))
    arrival_time = departure_time + timedelta(hours=random.randint(1, 5))

    # Get weather conditions
    temperature, visibility, wind_speed = generate_weather()

    # Flight status
    flight_status = generate_flight_status()

    # Append the flight data to the list
    flights.append({
        "FlightNumber": flight_number,
        "Airline": airline,
        "DepartureTime": departure_time.strftime("%Y-%m-%d %H:%M:%S"),
        "ArrivalTime": arrival_time.strftime("%Y-%m-%d %H:%M:%S"),
        "FlightStatus": flight_status,
        "Temperature(C)": temperature,
        "Visibility(km)": visibility,
        "WindSpeed(km/h)": wind_speed
    })

# Create a DataFrame
df = pd.DataFrame(flights)

# Save the DataFrame to a CSV file
df.to_csv("flights_data.csv", index=False)

# Display the first few rows of the generated dataset
print(df.head())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (ensure the dataset path is correct)
df = pd.read_csv("flights_data.csv")

# Display first few rows to understand the structure of the data
print("Dataset loaded successfully.")
print("First few rows of the data:")
print(df.head())

# 1. Preprocessing Data

# Convert 'DepartureTime' and 'ArrivalTime' to datetime if they are not already
df['DepartureTime'] = pd.to_datetime(df['DepartureTime'])
df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'])

# Extract useful time-related features
df['HourOfDay'] = df['DepartureTime'].dt.hour
df['DayOfWeek'] = df['DepartureTime'].dt.dayofweek  # 0=Monday, 6=Sunday

# One-hot encode categorical columns (Airline)
df = pd.get_dummies(df, columns=['Airline'], drop_first=True)

# Encode the 'FlightStatus' as 0 for "On-time" and 1 for "Delayed"
df['FlightStatus'] = df['FlightStatus'].apply(lambda x: 1 if x == 'Delayed' else 0)

# Drop columns that are not needed for the model
df.drop(columns=['FlightNumber', 'DepartureTime', 'ArrivalTime'], inplace=True)

# 2. Features and Target
X = df.drop(columns=['FlightStatus'])  # Features
y = df['FlightStatus']  # Target variable (0: On-time, 1: Delayed)

# 3. Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the model

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Evaluation Results:")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report
report = classification_report(y_test, y_pred, target_names=["On-time", "Delayed"], output_dict=True)

# Print key metrics: Precision, Recall, and F1-Score for each class
print("\nClassification Report (Precision, Recall, F1-Score):")
print(f"{'Class':<10}{'Precision':<12}{'Recall':<10}{'F1-Score':<10}")
for class_name in ["On-time", "Delayed"]:
    print(f"{class_name:<10}{report[class_name]['precision']:<12.2f}{report[class_name]['recall']:<10.2f}{report[class_name]['f1-score']:<10.2f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display confusion matrix with heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["On-time", "Delayed"], yticklabels=["On-time", "Delayed"])
plt.title("Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

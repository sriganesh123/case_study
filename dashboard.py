import xgboost as xgb
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Load models
churn_prediction = joblib.load('churn_prediction.pkl')
customer_segmentation = joblib.load('customer_segmentation.pkl')
lifetime_value_prediction = joblib.load('lifetime_value_prediction.pkl')

# Load the cluster labels mapping from the saved .pkl file
cluster_labels_mapping = joblib.load('cluster_labels_mapping.pkl')

# Helper function to set device on XGBoost models inside pipelines (set to CPU only)
def set_device_on_xgboost_model(pipeline):
    for name, step in pipeline.named_steps.items():
        if isinstance(step, xgb.XGBRegressor) or isinstance(step, xgb.XGBClassifier):
            step.set_params(tree_method='hist', nthread=3)  # Use single thread for inference
    return pipeline

# Set device for XGBoost models to ensure they use CPU
set_device_on_xgboost_model(lifetime_value_prediction)
set_device_on_xgboost_model(churn_prediction)

# Feature Engineering Function
def feature_engineering(inputs):
    inputs['Time_Since_Last_Renewal'] = (pd.to_datetime('2024-09-20') - pd.to_datetime(inputs['Renewal_or_Non-renewal_Date'])).dt.days
    inputs['Game_Attendance_Ratio'] = inputs['Games_Attended_Previous_Season'] / inputs['Total_Games_Previous_Season']
    inputs['Engagement_Score'] = inputs['Phone_Calls_Made_to_Customer_Last_90_Days_Before_Renewal_Date'] + \
                                 inputs['Live_Phone_Conversations_With_Customer_Last_90_Days_Before_Renewal_Date'] + \
                                 inputs['Emails_Sent_Last_90_Days_Before_Renewal_Date']
    
    inputs['Last_Engagement_Days'] = np.where(inputs['Emails_Sent_Last_90_Days_Before_Renewal_Date'] > 0,
                                              90 - inputs['Emails_Sent_Last_90_Days_Before_Renewal_Date'], 
                                              90)
    
    inputs['Revenue_Per_Game'] = (inputs['Average_Price_Paid_Per_Seat_Previous_Season'] * inputs['Total_Seats_Previous_Season']) / inputs['Games_Attended_Previous_Season']
    inputs['Games_Missed_Ratio'] = inputs['Games_Missed_Previous_Season'] / inputs['Total_Games_Previous_Season']
    inputs['Loyalty'] = (inputs['Tenure_As_Season_Ticket_Member'] > 3).astype(int)
    return inputs

# Preprocessing Functions for Each Model
def preprocess_churn(inputs):
    return inputs[['Engagement_Score', 'Loyalty', 'Revenue_Per_Game', 'Last_Engagement_Days']]

def preprocess_segmentation(inputs):
    return inputs[['Engagement_Score', 'Game_Attendance_Ratio', 'Revenue_Per_Game', 'Loyalty', 'Tenure_As_Season_Ticket_Member']]

def preprocess_lifetime_value(inputs):
    return inputs[['Revenue_Per_Game', 'Total_Seats_Previous_Season', 'Game_Attendance_Ratio', 'Engagement_Score', 'Tenure_As_Season_Ticket_Member']]

# Explanation helper function with Min/Max and specific value explanations
def explain_features(features, input_values):
    explanations = {
        'Engagement_Score': {
            'description': "Engagement Score: This score reflects how many times the customer interacted with the company, including phone calls and emails. A higher score indicates more engagement, potentially reducing churn risk.",
            'min': 0,
            'max': "No maximum",
            'specific_values': {
                '0': "No engagement from the customer (0 emails or phone calls)."
            }
        },
        'Loyalty': {
            'description': "Loyalty: This indicates whether the customer has been a season ticket member for more than 3 years. Long-tenure customers are more likely to renew.",
            'min': 0,
            'max': 1,
            'specific_values': {
                '0': "Customer has been a season ticket member for 3 years or less.",
                '1': "Customer has been a season ticket member for more than 3 years."
            }
        },
        'Revenue_Per_Game': {
            'description': "Revenue Per Game: This is the amount the customer spent per game in the previous season. Higher revenue per game generally indicates a higher value customer.",
            'min': 0,
            'max': "No maximum",
            'specific_values': {}
        },
        'Last_Engagement_Days': {
            'description': "Last Engagement Days: This shows how many days it has been since the last engagement with the customer (email, phone call). Longer periods without engagement may indicate a higher risk of churn.",
            'min': 0,
            'max': 90,
            'specific_values': {
                '0': "Customer has been engaged within the last day.",
                '90': "No engagement within the last 90 days."
            }
        },
        'Game_Attendance_Ratio': {
            'description': "Game Attendance Ratio: The ratio of games attended to the total number of games in the previous season. A higher ratio indicates stronger engagement with the games.",
            'min': 0,
            'max': 1,
            'specific_values': {
                '0': "Customer attended none of the games in the previous season.",
                '1': "Customer attended all the games in the previous season."
            }
        },
        'Tenure_As_Season_Ticket_Member': {
            'description': "Tenure as Season Ticket Member: The number of years the customer has been a season ticket member. Longer tenure is associated with higher loyalty.",
            'min': 0,
            'max': "No maximum",
            'specific_values': {}
        },
        'Time_Since_Last_Renewal': {
            'description': "Time Since Last Renewal: The number of days since the customer's last renewal or non-renewal. A longer time without engagement could signal a higher risk of churn.",
            'min': 0,
            'max': "No maximum",
            'specific_values': {}
        },
        'Total_Seats_Previous_Season': {
            'description': "Total Seats in Previous Season: The number of seats purchased by the customer in the previous season. More seats typically indicate higher engagement and value.",
            'min': 1,
            'max': "No maximum",
            'specific_values': {}
        }
    }

    # Create the explanation dictionary for the table
    table_explanations = {}
    for feature in features:
        explanation = explanations.get(feature, {})
        value = input_values[feature].iloc[0]  # Get the value of the feature from the input
        table_explanations[feature] = {
            'Value': value,
            'Explanation': explanation.get('description', ''),
            'Min': explanation.get('min', 'N/A'),
            'Max': explanation.get('max', 'N/A'),
            'Specific Meaning': explanation.get('specific_values', {}).get(str(value), '')
        }
    
    return table_explanations

# Streamlit Interface
st.title("Customer Model Prediction Dashboard")

# Option for user to either input data manually or upload a file
st.header("Choose Input Method")
input_method = st.radio("Select how you would like to provide customer data:", ("Manual Entry", "Upload CSV/Excel"))

if input_method == "Manual Entry":
    st.header("Customer Details Input")
    customer_id = st.text_input("Customer ID")
    renewal_date = st.date_input("Renewal or Non-renewal Date")
    emails_sent = st.number_input("Emails Sent Last 90 Days Before Renewal Date", min_value=0, value=0)
    phone_calls_made = st.number_input("Phone Calls Made Last 90 Days", min_value=0, value=0)
    live_phone_conversations = st.number_input("Live Phone Conversations Last 90 Days", min_value=0, value=0)
    games_attended = st.number_input("Games Attended Previous Season", min_value=0, value=0)
    total_games = st.number_input("Total Games Previous Season", min_value=1, value=1)  # Prevent division by 0
    average_price_paid = st.number_input("Average Price Paid Per Seat", min_value=0.0, value=0.0)
    total_seats = st.number_input("Total Seats Previous Season", min_value=1, value=1)
    games_missed = st.number_input("Games Missed Previous Season", min_value=0, value=0)
    tenure_season_ticket = st.number_input("Tenure As Season Ticket Member", min_value=0, value=0)

    # Collect all inputs into a DataFrame
    input_data = pd.DataFrame({
        'Renewal_or_Non-renewal_Date': [renewal_date],
        'Emails_Sent_Last_90_Days_Before_Renewal_Date': [emails_sent],
        'Phone_Calls_Made_to_Customer_Last_90_Days_Before_Renewal_Date': [phone_calls_made],
        'Live_Phone_Conversations_With_Customer_Last_90_Days_Before_Renewal_Date': [live_phone_conversations],
        'Games_Attended_Previous_Season': [games_attended],
        'Total_Games_Previous_Season': [total_games],
        'Average_Price_Paid_Per_Seat_Previous_Season': [average_price_paid],
        'Total_Seats_Previous_Season': [total_seats],
        'Games_Missed_Previous_Season': [games_missed],
        'Tenure_As_Season_Ticket_Member': [tenure_season_ticket]
    })

elif input_method == "Upload CSV/Excel":
    # Upload Option
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        # Load data from the file
        if uploaded_file.name.endswith('.csv'):
            input_data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            input_data = pd.read_excel(uploaded_file)

        # Display uploaded data for confirmation
        st.write("Uploaded data:")
        st.write(input_data)

        # Check if 'Customer_ID' column exists and allow selection
        if 'Customer_ID' in input_data.columns:
            customer_ids = input_data['Customer_ID'].unique()
            selected_customer_id = st.selectbox("Select Customer ID", customer_ids)

            # Filter data for the selected customer
            input_data = input_data[input_data['Customer_ID'] == selected_customer_id]

        else:
            st.warning("Customer_ID column not found in uploaded data.")
    else:
        st.warning("Please upload a file to proceed.")

if st.button("Run Models") and input_data is not None and not input_data.empty:
    
    # Feature Engineering
    input_data = feature_engineering(input_data)
    
    # Reset the index to avoid KeyError when accessing the first row
    input_data.reset_index(drop=True, inplace=True)

    # Churn Prediction
    churn_input = preprocess_churn(input_data)
    churn_result = churn_prediction.predict(churn_input)
    
    # Map the result for better readability
    result_mapping = {1: 'Renewed', 0: 'Not Renewed'}
    churn_prediction_result = result_mapping.get(churn_result[0], "Unknown")
    st.write(f"Churn Prediction Result: {churn_prediction_result}")
    
    # Display the features used for Churn Prediction
    churn_features = explain_features(churn_input.columns, churn_input)
    st.write("The Churn Prediction model based its prediction on the following factors:")
    churn_table = pd.DataFrame.from_dict(churn_features, orient='index')
    st.write(churn_table)

    # Customer Segmentation
    segmentation_input = preprocess_segmentation(input_data)
    segmentation_result = customer_segmentation.predict(segmentation_input)
    
    # Map the predicted segment to the corresponding cluster name
    predicted_segment = segmentation_result[0]
    cluster_name = cluster_labels_mapping.get(predicted_segment, "Unknown Segment")
    st.write(f"Customer Segment: {cluster_name}")

    # Display the features used for Customer Segmentation
    segmentation_features = explain_features(segmentation_input.columns, segmentation_input)
    st.write("The Customer Segmentation model based its prediction on the following factors:")
    segmentation_table = pd.DataFrame.from_dict(segmentation_features, orient='index')
    st.write(segmentation_table)

    # Lifetime Value Prediction
    lifetime_value_input = preprocess_lifetime_value(input_data)
    lifetime_value_result = lifetime_value_prediction.predict(lifetime_value_input)

    st.write(f"Predicted Lifetime Value: ${lifetime_value_result[0]:,.2f}")

    # Accessing values for Lifetime Value calculation
    revenue_per_game = input_data.loc[0, 'Revenue_Per_Game']
    games_attended_previous_season = input_data.loc[0, 'Games_Attended_Previous_Season']
    current_yearly_revenue = revenue_per_game * games_attended_previous_season

    st.write(f"Current Yearly Revenue (for season): ${current_yearly_revenue:.2f}")

    if current_yearly_revenue > 0:
        estimated_seasons = lifetime_value_result[0] / current_yearly_revenue
        st.write(f"This represents approximately {estimated_seasons:.2f} seasons of revenue.")
    else:
        st.write("Unable to estimate the number of seasons due to missing or zero revenue data.")

    # Display the features used for Lifetime Value Prediction
    lifetime_value_features = explain_features(lifetime_value_input.columns, lifetime_value_input)
    st.write("The Lifetime Value Prediction model based its prediction on the following factors:")
    
    lifetime_value_table = pd.DataFrame.from_dict(lifetime_value_features, orient='index')
    st.write(lifetime_value_table)


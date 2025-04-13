import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import os
import re


# Set up Streamlit page configuration
st.set_page_config(page_title="User Registration/Login", layout="centered")

# Change font style to Times New Roman and font size to 14
st.markdown(
    """
    <style>
        body {
            font-family: 'Times New Roman';
            font-size: 14px;
        }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {display: none;} /* Hide sidebar */
        [data-testid="stSidebarNavToggle"] {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Google Sheets Configuration
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), "hopeful-canto-451317-f9-501020437508.json")  # Ensure this JSON file is correct
REGISTRATION_DATA_SHEET_ID = "1FWXWAmK5B4NhYDAGi5H3twPNXg1f6vylHib4WJQR3Lw"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def connect_to_sheets(sheet_id):
    """Establish connection to a Google Sheet."""
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        st.error(f"Service account file '{SERVICE_ACCOUNT_FILE}' not found.")
        return None
    
    try:
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(sheet_id).sheet1
        return sheet
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return None

# Connect to Google Sheets
sheet = connect_to_sheets(REGISTRATION_DATA_SHEET_ID)

# Validation functions
def validate_name_input(new_value):
    """Allows only alphabets and spaces"""
    return bool(re.match(r"^[A-Za-z ]*$", new_value))

def is_valid_email(email):
    """Validates email format when clicking Register button"""
    return bool(re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email))

def validate_phone_input(new_value):
    """Allows only numbers, max 10 digits"""
    return new_value.isdigit() and len(new_value) <= 10

# Function to show login form
def show_login_form():
    st.header("Login")
    email = st.text_input("YOUR EMAIL", key="login_email")
    password = st.text_input("YOUR PASSWORD", type="password", key="login_password")
    if st.button("Log In"):
        login(email, password)

# Function to show signup form
def show_signup_form():
    st.header("Sign Up")
    name = st.text_input("YOUR USERNAME", key="name")
    email = st.text_input("YOUR EMAIL", key="email")
    phone = st.text_input("YOUR PHONE", key="phone")
    password = st.text_input("YOUR PASSWORD", type="password", key="password")
    confirm_password = st.text_input("REPEAT YOUR PASSWORD", type="password", key="confirm_password")
    
    if st.button("Sign Up"):
        save_to_google_sheets(name, email, phone, password, confirm_password)

# Function to save data to Google Sheets
def save_to_google_sheets(name, email, phone, password, confirm_password):
    if sheet is None:
        st.error("Could not connect to Google Sheets.")
        return
    
    if not name or not email or not phone or not password or not confirm_password:
        st.error("All fields are required!")
        return
    
    if not validate_name_input(name):
        st.error("Invalid Name! Only alphabets and spaces are allowed.")
        return
    
    if not is_valid_email(email):
        st.error("Invalid Email! Please enter a valid email address.")
        return
    
    if not validate_phone_input(phone):
        st.error("Invalid Phone Number! Only 10-digit numbers are allowed.")
        return
    
    if password != confirm_password:
        st.error("Passwords do not match!")
        return
    
    try:
        last_row = len(sheet.get_all_values()) + 1  # Finds the next empty row
        sheet.insert_row([name, email, phone, password], last_row)  # Include phone number
        st.success("Registration Successful!")
    except Exception as e:
        st.error(f"Failed to save data: {e}")

# Function to log in the user
def login(email, password):
    if sheet is None:
        st.error("Could not connect to Google Sheets.")
        return

    try:
        all_users = sheet.get_all_values()  # Fetch all records

        for row in all_users[1:]:  # Skip header row
            stored_name, stored_email, stored_phone, stored_password = row

            if email == stored_email and password == stored_password:
                st.success("Login Successful!")
                # Set session state to indicate the user is logged in
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.form = 'landing'  # Navigate to landing page
                st.rerun()  # This will refresh and show the landing page

                return

        st.error("Invalid email or password!")

    except Exception as e:
        st.error(f"Failed to retrieve data: {e}")



# Create navigation buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Go to Sign Up"):
        st.session_state.form = 'signup'

with col2:
    if st.button("Go to Log In"):
        st.session_state.form = 'login'

# Show forms dynamically based on session state
if 'form' not in st.session_state:
    st.session_state.form = 'login'  # Default to 'login' form

# If logged in, show the landing page
if 'logged_in' in st.session_state and st.session_state.logged_in:
    # Show the landing page content here
    st.title("Landing Page")
    st.write(f"Welcome, {st.session_state.user_email}!")
    # st.session_state.page == "landing_page"
    st.switch_page("Landing_page.py")    
    # Add your landing page functionality here
    # For example, buttons or further navigation options
    
else:
    if st.session_state.form == 'login':
        show_login_form()
    elif st.session_state.form == 'signup':
        show_signup_form()

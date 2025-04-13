# import streamlit as st
# from PIL import Image

# # ✅ Ensure session state variables are initialized
# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False
# if "user_email" not in st.session_state:
#     st.session_state.user_email = ""
# if "page" not in st.session_state:
#     st.session_state.page = "landing_page"  # Default to landing page

# # ✅ Redirect users to login if they are not logged in
# if not st.session_state.logged_in:
#     st.switch_page("pages/registration.py")  # Redirect to the registration/login page
#     st.stop()  # Stop execution to prevent the rest of the code from running

# # ✅ Set page configuration
# st.set_page_config(page_title="Landing Page", layout="wide")

# # ✅ Hide Sidebar (CSS Trick)
# st.markdown(
#     """
#     <style>
#         [data-testid="stSidebar"] {display: none !important;} /* Hide sidebar */
#         [data-testid="collapsedControl"] {display: none !important;} /* Hide the toggle button */
#         .image-container {
#             display: flex;
#             justify-content: center;
#             align-items: center;
#             flex-direction: column;
#             width: 300px; /* Ensuring same width */
#         }
#         .button-container {
#             width: 300px;
#             display: flex;
#             justify-content: center;
#             margin-top: 10px;          
#         }
#         .button-container button {
#             margin-left: 50px; /* Move button slightly to the right */
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # ✅ Title
# st.markdown("<h1 style='text-align: center;'>Parkinson Disease Prediction</h1>", unsafe_allow_html=True)
# st.write(f"Welcome, **{st.session_state.user_email}**!")

# # ✅ Sign Out Button
# col1, col2 = st.columns([8, 1])  # Create two columns for layout
# with col2:
#     signout_button = st.button("Sign Out")
#     if signout_button:
#         st.session_state.logged_in = False
#         del st.session_state["user_email"]
#         st.switch_page("pages/registration.py")
#         st.rerun()  

# # ✅ Load images using PIL
# def load_image(path):
#     try:
#         img = Image.open(path)
#         img = img.resize((300, 300), Image.Resampling.LANCZOS)  # Fixed width and height
#         return img
#     except Exception as e:
#         st.error(f"Error loading {path}: {e}")
#         return None

# # ✅ Load images
# img1 = load_image("report_img.png")
# img2 = load_image("brain_img.png")

# # ✅ Display images and buttons in aligned columns
# col1, col2 = st.columns([1, 1])

# with col1:
#     if img1:
#         st.markdown('<div class="image-container">', unsafe_allow_html=True)
#         st.image(img1)
#         st.markdown('</div>', unsafe_allow_html=True)

#     # ✅ Move Button Slightly Right
#     st.markdown('<div class="button-container">', unsafe_allow_html=True)
#     btn1 = st.button("Go to Tabular Data", key="btn1")
#     if btn1:
#         st.switch_page("pages/tabular_data.py")
#     st.markdown('</div>', unsafe_allow_html=True)

# with col2:
#     if img2:
#         st.markdown('<div class="image-container">', unsafe_allow_html=True)
#         st.image(img2)
#         st.markdown('</div>', unsafe_allow_html=True)

#     # ✅ Move Button Slightly Right
#     st.markdown('<div class="button-container">', unsafe_allow_html=True)
#     btn2 = st.button("Go to Prediction Page", key="btn2")
#     if btn2:
#         st.switch_page("pages/GUI_NEW.py")
#     st.markdown('</div>', unsafe_allow_html=True)



import streamlit as st
from PIL import Image
import gspread
from google.oauth2.service_account import Credentials
import os

# ✅ Page configuration
st.set_page_config(page_title="Landing Page", layout="wide")

# === Google Sheets Configuration ===
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), "regal-station-452514-t8-42ab438bf0cc.json")
SHEET_ID = "1FWXWAmK5B4NhYDAGi5H3twPNXg1f6vylHib4WJQR3Lw"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# ✅ Function to connect and fetch data from Google Sheets
@st.cache_resource
def get_username_by_email(email):
    try:
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID).sheet1
        data = sheet.get_all_records()

        # Find the user by email (case-insensitive)
        for record in data:
            if record.get("EMAILID", "").strip().lower() == email.strip().lower():
                return record.get("NAME")
        return None  # If email not found
    except Exception as e:
        st.error(f"⚠ Error fetching user data: {e}")
        return None

# ✅ Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "username" not in st.session_state:
    st.session_state.username = ""

# ✅ Fetch username after login
if st.session_state.logged_in and not st.session_state.get("username"):
    username = get_username_by_email(st.session_state.user_email)
    if username:
        st.session_state.username = username
    else:
        st.session_state.username = st.session_state.user_email  # Fallback if username is not found


# ✅ Redirect to login if not logged in
if not st.session_state.logged_in:
    st.switch_page("pages/registration.py")
    st.stop()



# ✅ Hide Sidebar
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {display: none !important;}
        [data-testid="collapsedControl"] {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ✅ Display Title and Welcome Message
st.markdown("<h1 style='text-align: center;'>Parkinson Disease Prediction</h1>", unsafe_allow_html=True)
st.write(f"Welcome, **{st.session_state.username or st.session_state.user_email}**!")

# ✅ Sign Out Button
col1, col2 = st.columns([8, 1])
with col2:
    if st.button("Sign Out"):
        st.session_state.logged_in = False
        st.session_state.user_email = ""
        st.session_state.username = ""
        st.switch_page("pages/registration.py")
        st.rerun()

# ✅ Load Images Function
def load_image(path):
    try:
        img = Image.open(path)
        img = img.resize((300, 300), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None

# ✅ Load and Display Images with Buttons
img1 = load_image("report_img.png")
img2 = load_image("brain_img.png")

col1, col2 = st.columns(2)

with col1:
    if img1:
        st.image(img1)
    if st.button("Go to Tabular Data", key="btn1"):
        st.switch_page("pages/tabular_data.py")

with col2:
    if img2:
        st.image(img2)
    if st.button("Go to Prediction Page", key="btn2"):
        st.switch_page("pages/GUI_NEW.py")
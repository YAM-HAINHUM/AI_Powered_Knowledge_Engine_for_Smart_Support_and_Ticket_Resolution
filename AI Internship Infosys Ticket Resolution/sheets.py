import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
from pathlib import Path

# ----------------- CONFIG -----------------
# Path to your Google service account JSON key
CREDENTIALS_FILE = r"C:\Users\Admin\Desktop\AI Internship Infosys\credentials.json"
SHEET_NAME = "AI INTERNSHIP INFOSYS"  # Name of your Google Sheet

# ----------------- SETUP -----------------
if not Path(CREDENTIALS_FILE).exists():
    raise FileNotFoundError(f"{CREDENTIALS_FILE} not found. Place your service account key here.")

# Define the scope for Google Sheets API
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

# Authorize the client
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
client = gspread.authorize(creds)

# Open the Google Sheet
try:
    sheet = client.open("AI INTERNSHIP INFOSYS").sheet1
except gspread.SpreadsheetNotFound:
    raise ValueError(
        f"Spreadsheet 'AI INTERNSHIP INFOSYS' not found. Make sure it exists and is shared with your service account email."
    )

# ----------------- FUNCTION -----------------
def save_query_to_sheets(query: str, response: str):
    """
    Save a query and its response to Google Sheets with a timestamp.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([timestamp, query, response])
    print("✅ Query saved to Google Sheets!")

# ----------------- TEST -----------------
if __name__ == "__main__":
    # Example test
    test_query = "What is the SLA for high priority tickets?"
    test_response = "The SLA is 4 hours for high priority tickets."
    save_query_to_sheets(test_query, test_response)

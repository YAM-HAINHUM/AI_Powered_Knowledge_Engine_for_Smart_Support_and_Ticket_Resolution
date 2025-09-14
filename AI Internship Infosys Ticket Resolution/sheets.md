Step1:

Go to Google Cloud Console
.

Create a new project (or use existing).

Enable the Google Sheets API and Google Drive API.

Create credentials → Service Account → download the JSON key file (e.g., credentials.json).

Share your Google Sheet with the service account email (something like your-service@project.iam.gserviceaccount.com) and give Editor access.


Step 2:

pip install gspread oauth2client pandas


import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# Define the scope
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]

# Load credentials
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)

# Authorize client
client = gspread.authorize(creds)

# Open the sheet by name
sheet = client.open("MyDatabaseSheet").sheet1  # first worksheet

# Read all records into a list of dicts
data = sheet.get_all_records()

# Convert to pandas DataFrame
df = pd.DataFrame(data)

print(df.head())


# Update a single cell
sheet.update_cell(2, 3, "Hello World")  # row=2, col=3

# Append a row
sheet.append_row(["2025-08-25", "John Doe", 100])




Alternative

pip install pandas-gbq

from google.oauth2 import service_account
import pandas_gbq

credentials = service_account.Credentials.from_service_account_file("credentials.json")
query = "SELECT * FROM `project.dataset.table`"
df = pandas_gbq.read_gbq(query, project_id="your-project-id", credentials=credentials)

import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import uuid
import pandas as pd

# Google Sheets setup
SHEET_ID = "13IfoSOZtEdqQ1mDhQrPHtEO06OFMOSs0M4TWy5PrG6U"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

# Ensure headers exist
HEADERS = [
    "Ticket ID",
    "Ticket Content",
    "Ticket Timestamp",
    "Ticket By",
    "Ticket Raised By",
    "Ticket Category",
    "Ticket Problem",
    "Ticket Solution",
    "Ticket Status"
]

if sheet.row_values(1) != HEADERS:
    sheet.insert_row(HEADERS, 1)

# Function to categorize ticket
def categorize_ticket(content: str) -> str:
    content = content.lower()
    if "not delivered" in content or "didn't receive" in content or "late" in content:
        return "Delivery Issue"
    elif "defective" in content or "broken" in content or "damaged" in content:
        return "Product Quality"
    elif "refund" in content or "return" in content or "exchange" in content:
        return "Refund/Return"
    elif "payment" in content or "charged" in content or "billing" in content:
        return "Payment Issue"
    elif "account" in content or "login" in content or "password" in content:
        return "Account Issue"
    else:
        return "General Inquiry"

# Create new ticket
def create_ticket():
    ticket_content = input("Enter Ticket Content: ").strip()
    ticket_problem = input("Enter Ticket Problem: ").strip()
    ticket_solution = input("Enter Ticket Solution: ").strip()
    ticket_by = input("Enter Ticket By (customer name/email): ").strip()
    raised_by = input("Ticket Raised By (agent/system): ").strip()
    ticket_status = input("Enter Ticket Status (Open/In Progress/Closed): ").strip()

    ticket_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    category = categorize_ticket(ticket_content)

    ticket_data = [
        ticket_id,
        ticket_content,
        timestamp,
        ticket_by,
        raised_by,
        category,
        ticket_problem,
        ticket_solution,
        ticket_status
    ]

    sheet.append_row(ticket_data)
    print(f"\n✅ Ticket {ticket_id} created and stored successfully.\n")

# View all tickets
def view_tickets():
    records = sheet.get_all_records()
    if not records:
        print("\n⚠️ No tickets found.\n")
        return
    df = pd.DataFrame(records)
    print("\n📋 All Tickets:\n")
    print(df.to_string(index=False))
    print()

# Update ticket status
def update_ticket_status():
    ticket_id = input("Enter Ticket ID to update: ").strip()
    records = sheet.get_all_records()

    for idx, record in enumerate(records, start=2):  # row 2 onwards
        if record["Ticket ID"] == ticket_id:
            new_status = input("Enter new status (Open/In Progress/Closed): ").strip()
            sheet.update_cell(idx, HEADERS.index("Ticket Status") + 1, new_status)
            print(f"\n✅ Ticket {ticket_id} status updated to {new_status}.\n")
            return

    print("\n⚠️ Ticket ID not found.\n")

# Main menu
def main_menu():
    while True:
        print("===== Amazon Ticket Management =====")
        print("1. Create Ticket")
        print("2. View All Tickets")
        print("3. Update Ticket Status")
        print("4. Exit")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            create_ticket()
        elif choice == "2":
            view_tickets()
        elif choice == "3":
            update_ticket_status()
        elif choice == "4":
            print("👋 Exiting system. Goodbye!")
            break
        else:
            print("\n⚠️ Invalid choice, try again.\n")

if __name__ == "__main__":
    main_menu()




"""# Instead of asking for input every time, we provide default/sample values
ticket_content = "Order not delivered yet"
ticket_problem = "Delivery Issue"
ticket_solution = "Rescheduled delivery and escalated to logistics team"
ticket_by = "customer@example.com"
raised_by = "System"
ticket_status = "Open"

# With default fallback input (press enter to use default)
ticket_content = input("Enter Ticket Content (default: Order not delivered yet): ").strip() or "Order not delivered yet"
ticket_problem = input("Enter Ticket Problem (default: Delivery Issue): ").strip() or "Delivery Issue"
ticket_solution = input("Enter Ticket Solution (default: Rescheduled delivery and escalated to logistics team): ").strip() or "Rescheduled delivery and escalated to logistics team"
ticket_by = input("Enter Ticket By (default: customer@example.com): ").strip() or "customer@example.com"
raised_by = input("Ticket Raised By (default: System): ").strip() or "System"
ticket_status = input("Enter Ticket Status (default: Open): ").strip() or "Open"
"""
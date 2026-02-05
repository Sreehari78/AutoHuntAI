import os
import json
import asyncio
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.sessions import StringSession
import gspread
import google.generativeai as genai
from itertools import cycle

# Load environment variables
load_dotenv()


class APIKeyRotator:
    """
    Manages multiple Gemini API keys and rotates through them when rate limits are hit.
    """
    def __init__(self):
        # Load all API keys from environment (supports up to 3 keys)
        self.api_keys = []
        for i in range(1, 4):  # API keys 1-3
            key = os.getenv(f'GEMINI_API_KEY_{i}')
            if key:
                self.api_keys.append(key)
        
        if not self.api_keys:
            raise ValueError("No API keys found! Please set GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc. in .env")
        
        self.key_cycle = cycle(self.api_keys)
        self.current_key = next(self.key_cycle)
        self.current_key_index = 0
        self.request_counts = {i: 0 for i in range(len(self.api_keys))}
        
        print(f"✅ Loaded {len(self.api_keys)} API key(s) for rotation")
    
    def get_current_key(self):
        """Get the current API key"""
        return self.current_key
    
    def rotate_key(self):
        """Switch to the next API key"""
        self.current_key = next(self.key_cycle)
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        print(f"  🔄 Rotated to API key #{self.current_key_index + 1}")
        return self.current_key
    
    def get_model(self):
        """Get a Gemini model instance with the current API key"""
        genai.configure(api_key=self.current_key)
        model_name = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash-exp')
        return genai.GenerativeModel(model_name)
    
    def record_request(self):
        """Record that a request was made with the current key"""
        self.request_counts[self.current_key_index] += 1


# Initialize API key rotator
api_rotator = APIKeyRotator()

# Google Sheets setup
google_creds_dict = json.loads(os.environ['GOOGLE_CREDENTIALS'])
gc = gspread.service_account_from_dict(google_creds_dict)
sheet_name = os.environ.get('SHEET_NAME', 'JobSheetsAI')

# Telegram setup
api_id = int(os.environ['API_ID'])
api_hash = os.environ['API_HASH']
session_string = os.environ.get('SESSION_STRING', '')
channel_links = os.environ.get('CHANNEL_LINKS', '').split(',')
channel_links = [link.strip() for link in channel_links if link.strip()]

client = TelegramClient(StringSession(session_string), api_id, api_hash)

# Column headers for the sheet
SHEET_HEADERS = [
    "Datetime",
    "Company Name", 
    "Role",
    "Compensation (CTC)",
    "YEARS OF EXPERIENCE",
    "PASSOUT YEAR",
    "Application Link",
    "Original Message",
    "Status"
]


def parse_job_with_gemini(message_text, message_date, max_retries=3):
    """
    Use Gemini AI to extract structured job information from a message.
    Automatically rotates API keys when rate limits are hit.
    
    Args:
        message_text: The message text to parse
        message_date: The message date
        max_retries: Maximum number of retries with key rotation (default: 3)
        
    Returns:
        dict: Extracted job information or None if parsing failed
    """
    if not message_text or len(message_text.strip()) < 10:
        return None
    
    prompt = f"""You are a job posting parser. Extract the following information from the job posting below.
Return ONLY a valid JSON object with these exact fields (use null for missing information):

{{
    "company_name": "Company name",
    "role": "Job role/title",
    "compensation": "Salary/CTC information (e.g., '10-15 LPA')",
    "years_of_experience": "Years of experience required (e.g., '0-2', '3+', 'Freshers')",
    "passout_year": "Target graduation year(s) (e.g., '2024', '2023-2025', 'All')",
    "application_link": "Application URL if present"
}}

Rules:
- Extract data as accurately as possible
- For years_of_experience: look for phrases like "0-2 years", "Freshers", "1+ years", "Experience: 2 years"
- For passout_year: look for phrases like "2024 graduates", "2023, 2024, 2025 grads", "All batches"
- For compensation: include currency and format (e.g., "10-15 LPA", "₹8-12 LPA")
- For application_link: extract the actual application URL, not promotional links
- If multiple URLs exist, pick the one that looks like a job application link
- Set field to null if information is not found

Job Posting:
{message_text[:500]}

JSON Output:"""

    for attempt in range(max_retries):
        try:
            # Get model with current API key
            gemini_model = api_rotator.get_model()
            api_rotator.record_request()
            
            response = gemini_model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean up markdown code blocks if present
            if result_text.startswith('```'):
                result_text = result_text.split('```')[1]
                if result_text.startswith('json'):
                    result_text = result_text[4:]
                result_text = result_text.strip()
            
            # Parse JSON
            job_data = json.loads(result_text)
            
            # Add datetime and original message
            job_data['datetime'] = message_date.strftime('%Y-%m-%d %H:%M:%S')
            job_data['original_message'] = message_text
            job_data['status'] = '📋 InQueue'
            
            return job_data
        
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a rate limit error
            if '429' in error_str or 'resource_exhausted' in error_str or 'quota' in error_str or 'rate limit' in error_str:
                if attempt < max_retries - 1:
                    print(f"  ⚠ Rate limit hit! Rotating API key...")
                    api_rotator.rotate_key()
                    time.sleep(2)  # Small delay before retry
                    continue
                else:
                    print(f"  ❌ All API keys exhausted: {str(e)[:100]}")
                    return None
            else:
                # Other errors - don't retry
                print(f"  ⚠ Gemini parsing error: {str(e)[:100]}")
                return None
    
    return None


def get_or_create_sheet(sheet_name):
    """
    Get or create the Google Sheet with proper headers.
    
    Args:
        sheet_name: Name of the sheet
        
    Returns:
        gspread.Worksheet: The worksheet object
    """
    try:
        # Try to open existing spreadsheet
        spreadsheet = gc.open(sheet_name)
        worksheet = spreadsheet.sheet1
        
        # Check if headers exist
        existing_headers = worksheet.row_values(1)
        if existing_headers != SHEET_HEADERS:
            print(f"  📝 Setting headers...")
            worksheet.clear()
            worksheet.append_row(SHEET_HEADERS)
            
    except gspread.SpreadsheetNotFound:
        # Create new spreadsheet
        print(f"  📝 Creating new spreadsheet '{sheet_name}'...")
        spreadsheet = gc.create(sheet_name)
        worksheet = spreadsheet.sheet1
        worksheet.append_row(SHEET_HEADERS)
    
    return worksheet


def is_duplicate(worksheet, application_link):
    """
    Check if a job posting already exists in the sheet.
    
    Args:
        worksheet: The Google Sheet worksheet
        application_link: The application link to check
        
    Returns:
        bool: True if duplicate exists, False otherwise
    """
    if not application_link:
        return False
    
    try:
        # Get all application links from column G (index 7)
        all_links = worksheet.col_values(7)  # Application Link column
        return application_link in all_links
    except:
        return False


def upload_jobs_to_sheet(jobs_data):
    """
    Upload job data to Google Sheets with duplicate detection.
    
    Args:
        jobs_data: List of job dictionaries
        
    Returns:
        tuple: (total_jobs, uploaded_jobs, duplicate_jobs)
    """
    if not jobs_data:
        return 0, 0, 0
    
    print(f"\n📊 Uploading to Google Sheets...")
    worksheet = get_or_create_sheet(sheet_name)
    
    uploaded = 0
    duplicates = 0
    
    for job in jobs_data:
        # Check for duplicates
        if is_duplicate(worksheet, job.get('application_link')):
            duplicates += 1
            continue
        
        # Prepare row data matching SHEET_HEADERS order
        row_data = [
            job.get('datetime', ''),
            job.get('company_name', ''),
            job.get('role', ''),
            job.get('compensation', ''),
            job.get('years_of_experience', ''),
            job.get('passout_year', ''),
            job.get('application_link', ''),
            job.get('original_message', ''),
            job.get('status', 'InQueue')
        ]
        
        worksheet.append_row(row_data)
        uploaded += 1
    
    return len(jobs_data), uploaded, duplicates


async def main():
    """Main function to fetch, parse, and upload job postings"""
    async with client:
        await client.start()
        print(f"✅ Connected to Telegram!")
        
        # Calculate 1 hour ago
        now = datetime.now(timezone.utc)
        cutoff_time = now - timedelta(hours=1)
        
        print(f"⏰ Fetching messages from the last 1 hour")
        print(f"   Cutoff: {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
        
        all_jobs = []
        total_messages = 0
        
        for channel_link in channel_links:
            try:
                print(f"{'='*80}")
                print(f"📢 Channel: {channel_link}")
                print(f"{'='*80}")
                
                # Get channel entity
                channel = await client.get_entity(channel_link)
                print(f"   Name: {channel.title}")
                
                # Fetch messages from last 24 hours
                messages_in_range = []
                async for message in client.iter_messages(channel):
                    if message.date >= cutoff_time:
                        messages_in_range.append(message)
                    else:
                        break
                
                print(f"   📥 Found {len(messages_in_range)} messages")
                total_messages += len(messages_in_range)
                
                # Parse each message with Gemini
                for i, msg in enumerate(messages_in_range, 1):
                    if msg.text:
                        print(f"   🤖 Parsing message {i}/{len(messages_in_range)}...", end=" ")
                        job_data = parse_job_with_gemini(msg.text, msg.date)
                        
                        if job_data:
                            all_jobs.append(job_data)
                            print(f"✅ {job_data.get('company_name', 'Unknown')} - {job_data.get('role', 'Unknown')}")
                        else:
                            print("⏭ Skipped (no job data)")
                
                print()
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}\n")
                continue
        
        # Upload to Google Sheets
        print(f"\n{'='*80}")
        print(f"📤 UPLOAD SUMMARY")
        print(f"{'='*80}")
        print(f"   Total messages fetched: {total_messages}")
        print(f"   Jobs parsed by Gemini: {len(all_jobs)}")
        
        if all_jobs:
            total, uploaded, duplicates = upload_jobs_to_sheet(all_jobs)
            print(f"   ✅ New jobs uploaded: {uploaded}")
            print(f"   🔄 Duplicates skipped: {duplicates}")
            print(f"\n🎉 Done! Check your Google Sheet: '{sheet_name}'")
        else:
            print(f"   ℹ No jobs to upload")


if __name__ == "__main__":
    asyncio.run(main())

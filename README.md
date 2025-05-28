# 🤖 Personal Assistant Bot

A fuzzy input processor that converts natural language requests into structured JSON responses with intent classification, entity extraction, and web search capabilities.

## Features

- **Intent Classification**: Categorizes requests into dining, travel, gifting, cab booking, or other
- **Entity Extraction**: Extracts relevant details like dates, locations, party size, budget, etc.
- **Confidence Scoring**: Provides confidence score for categorization accuracy
- **Follow-up Questions**: Generates questions for missing critical information
- **Web Search Integration**: For "other" category requests, performs web search to provide additional context
- **Interactive UI**: Streamlit-based interface for easy testing and visualization

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Google API Key (from Google AI Studio)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd personal-assistant-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create requirements.txt file** (if not provided):
   ```txt
   streamlit==1.28.0
   langchain-google-genai==1.0.1
   langchain==0.1.0
   pydantic==2.5.0
   duckduckgo-search==3.9.6
   ```

4. **Get Google API Key**
   - Visit [Google AI Studio](https://aistudio.google.com)
   - Create a new API key
   - Keep it secure for use in the application

### Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Access the interface**
   - Open your browser and go to `http://localhost:8501`
   - Enter your Google API Key in the sidebar
   - Start making requests!

## Usage Examples

### Standard Category Requests

#### Dining Request
**Input:** "Need a sunset-view table for two tonight; gluten-free menu a must"

**Expected Output:**
- Intent: dining
- Entities: party_size: 2, date: today, dietary_restrictions: gluten-free, ambiance: sunset view
- Follow-up questions about specific time

#### Travel Request
**Input:** "Planning a trip to Paris for 4 people next month"

**Expected Output:**
- Intent: travel
- Entities: destination: Paris, travelers_count: 4
- Follow-up questions about specific dates, budget, accommodation

#### Gifting Request
**Input:** "Need a gift for my mom's birthday under $100"

**Expected Output:**
- Intent: gifting
- Entities: recipient: mom, occasion: birthday, budget: $100
- Follow-up questions about preferences, delivery date

#### Cab Booking Request
**Input:** "Book a cab to airport tomorrow morning for 3 passengers"

**Expected Output:**
- Intent: cab_booking
- Entities: destination: airport, date: tomorrow, time: morning, passengers_count: 3
- Follow-up questions about pickup location, specific time

### Other Category Requests (with Web Search)

#### Government Services
**Input:** "How to update Aadhar address online"

**Expected Output:**
- Intent: other
- Web search results with official links and procedures

#### Technical Queries
**Input:** "Best practices for Python exception handling"

**Expected Output:**
- Intent: other
- Web search results with technical articles and documentation

## API Structure

### Input
Natural language text request

### Output JSON Schema
```json
{
  "intent_category": "string",
  "key_entities": {
    "key": "value"
  },
  "confidence_score": 0.85,
  "follow_up_questions": [
    "string"
  ],
  "web_search_results": [
    {
      "title": "string",
      "url": "string", 
      "snippet": "string"
    }
  ]
}
```

## Key Components

### Intent Categories

- **dining**: Restaurant reservations, food-related requests
- **travel**: Trip planning, flight/hotel bookings
- **gifting**: Gift recommendations and purchases
- **cab_booking**: Transportation requests
- **other**: General queries requiring web search

### Entity Types by Category

**Dining**
- date, time, location, cuisine, party_size, dietary_restrictions, budget, ambiance

**Travel**
- destination, departure_date, return_date, travelers_count, budget, accommodation_type, transport_mode

**Gifting**
- recipient, occasion, budget, preferences, delivery_date, gift_type

**Cab Booking**
- pickup_location, destination, date, time, passengers_count, cab_type, budget

### Confidence Scoring

- **0.9-1.0**: Very clear intent with specific details
- **0.7-0.9**: Clear intent but missing some details
- **0.5-0.7**: Somewhat ambiguous but can categorize
- **0.3-0.5**: Very ambiguous
- **0.0-0.3**: Cannot determine intent

## Error Handling

- Fallback parsing when LLM fails
- Graceful handling of API failures
- Web search error recovery
- Input validation and sanitization

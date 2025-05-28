import streamlit as st
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import re
from dataclasses import dataclass
from enum import Enum
import numpy as np

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Web search
from duckduckgo_search import DDGS

class IntentCategory(str, Enum):
    DINING = "dining"
    TRAVEL = "travel"
    GIFTING = "gifting"
    CAB_BOOKING = "cab_booking"
    OTHER = "other"

class AssistantResponse(BaseModel):
    intent_category: str = Field(description="The main category of the user request")
    key_entities: Dict[str, Any] = Field(description="Extracted entities from the request")
    confidence_score: float = Field(description="Confidence score from log probabilities")
    category_probabilities: Dict[str, float] = Field(description="Log probabilities for each category")
    follow_up_questions: List[str] = Field(description="Questions to ask for missing information")
    web_search_results: Optional[List[Dict[str, str]]] = Field(default=None, description="Web search results for non-standard requests")

@dataclass
class PersonalAssistant:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1
        )
        self.parser = PydanticOutputParser(pydantic_object=AssistantResponse)
        self.categories = ["dining", "travel", "gifting", "cab_booking", "other"]
        
    def create_classification_prompt(self) -> str:
        return """You are a personal assistant that classifies user requests into specific categories.

Your task is to:
1. Classify the request into ONE of these categories: dining, travel, gifting, cab_booking, or other
2. Respond with ONLY the category name (single word)

Categories:
- dining: Restaurant reservations, food orders, meal planning
- travel: Trip planning, flights, hotels, vacation bookings
- gifting: Gift suggestions, present shopping, occasion planning
- cab_booking: Taxi/cab booking, ride sharing, transportation
- other: Everything else that doesn't fit above categories

Examples:
Input: "Need a sunset-view table for two tonight"
Output: dining

Input: "Planning a trip to Paris for 4 people"
Output: travel

Input: "Need a gift for my mom's birthday"
Output: gifting

Input: "Book a cab to airport"
Output: cab_booking

Input: "How to update Aadhar address"
Output: other

Respond with ONLY the category name."""

    def get_category_probabilities(self, user_input: str) -> Dict[str, float]:
        """Get log probabilities for each category using multiple classification attempts"""
        classification_prompt = self.create_classification_prompt()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", classification_prompt),
            ("human", "{input}")
        ])
        
        chain = prompt | self.llm
        
        # Initialize probability tracking
        category_scores = {cat: 0.0 for cat in self.categories}
        num_attempts = 10  # More attempts for better distribution
        
        try:
            # Multiple classification attempts to estimate probabilities
            for _ in range(num_attempts):
                response = chain.invoke({"input": user_input})
                predicted_category = response.content.strip().lower()
                
                if predicted_category in self.categories:
                    category_scores[predicted_category] += 1.0
                else:
                    # If response doesn't match categories, assign to 'other'
                    category_scores["other"] += 1.0
            
            # Convert counts to probabilities with smoothing
            total_attempts = num_attempts
            smoothing_factor = 0.1  # Add some smoothing to avoid zero probabilities
            
            # Apply Laplace smoothing
            for category in self.categories:
                category_scores[category] += smoothing_factor
            
            total_with_smoothing = total_attempts + (len(self.categories) * smoothing_factor)
            
            # Calculate probabilities
            probabilities = {}
            for category in self.categories:
                prob = category_scores[category] / total_with_smoothing
                probabilities[category] = prob
            
            # Convert to log probabilities
            log_probs = {}
            for category in self.categories:
                log_probs[category] = np.log(probabilities[category])
            
            return log_probs
            
        except Exception as e:
            st.error(f"Error getting category probabilities: {str(e)}")
            # Return uniform log probabilities as fallback
            uniform_prob = 1.0 / len(self.categories)
            return {cat: np.log(uniform_prob) for cat in self.categories}
    
    def create_entity_extraction_prompt(self) -> str:
        return """You are a personal assistant that extracts key entities from user requests.

Based on the category, extract relevant entities:

ENTITY EXTRACTION GUIDELINES:
- DINING: date, time, location, cuisine, party_size, dietary_restrictions, budget, ambiance
- TRAVEL: destination, departure_date, return_date, travelers_count, budget, accommodation_type, transport_mode
- GIFTING: recipient, occasion, budget, preferences, delivery_date, gift_type
- CAB_BOOKING: pickup_location, destination, date, time, passengers_count, cab_type, budget
- OTHER: extract any relevant details

FOLLOW-UP QUESTIONS:
Generate questions for missing critical information:
- Dining: If no time/date, ask when. If no party size, ask how many people.
- Travel: If no travelers count, ask how many people. If no dates, ask when.
- Gifting: If no budget, ask for price range. If no occasion, ask what's the occasion.
- Cab: If no destination, ask where to. If no pickup, ask pickup location.

Respond in JSON format with:
{{
    "key_entities": {{}},
    "follow_up_questions": []
}}"""

    def extract_entities_and_questions(self, user_input: str, category: str) -> Dict[str, Any]:
        """Extract entities and generate follow-up questions"""
        entity_prompt = self.create_entity_extraction_prompt()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", entity_prompt),
            ("human", "Category: {category}\nUser request: {input}\n\nExtract entities and generate follow-up questions in JSON format.")
        ])
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({"input": user_input, "category": category})
            content = response.content
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_data = json.loads(json_str)
                return {
                    "key_entities": parsed_data.get("key_entities", {}),
                    "follow_up_questions": parsed_data.get("follow_up_questions", [])
                }
            else:
                return self._fallback_entity_extraction(user_input, category)
                
        except Exception as e:
            st.error(f"Error extracting entities: {str(e)}")
            return self._fallback_entity_extraction(user_input, category)
    
    def _fallback_entity_extraction(self, user_input: str, category: str) -> Dict[str, Any]:
        """Fallback entity extraction using regex patterns"""
        entities = {}
        questions = []
        
        if category == "dining":
            entities = self._extract_dining_entities(user_input)
            questions = ["What time would you prefer?", "How many people will be dining?"]
        elif category == "travel":
            entities = self._extract_travel_entities(user_input)
            questions = ["When would you like to travel?", "How many travelers?"]
        elif category == "gifting":
            entities = self._extract_gifting_entities(user_input)
            questions = ["What's your budget range?", "What's the occasion?"]
        elif category == "cab_booking":
            entities = self._extract_cab_entities(user_input)
            questions = ["What's your destination?", "When do you need the ride?"]
        else:
            entities = {"query": user_input}
            questions = ["Could you provide more details about what you need?"]
        
        return {
            "key_entities": entities,
            "follow_up_questions": questions
        }
    
    def _extract_dining_entities(self, text: str) -> Dict[str, Any]:
        entities = {}
        
        party_match = re.search(r'(\d+)\s*(people|person|pax)', text.lower())
        if party_match:
            entities['party_size'] = int(party_match.group(1))
        
        if 'tonight' in text.lower():
            entities['date'] = 'today'
        
        if 'gluten-free' in text.lower():
            entities['dietary_restrictions'] = ['gluten-free']
        
        if 'sunset' in text.lower():
            entities['ambiance'] = 'sunset view'
            
        return entities
    
    def _extract_travel_entities(self, text: str) -> Dict[str, Any]:
        entities = {}
        
        traveler_match = re.search(r'(\d+)\s*(people|person|traveler)', text.lower())
        if traveler_match:
            entities['travelers_count'] = int(traveler_match.group(1))
            
        return entities
    
    def _extract_gifting_entities(self, text: str) -> Dict[str, Any]:
        entities = {}
        
        budget_match = re.search(r'\$(\d+)', text)
        if budget_match:
            entities['budget'] = f"${budget_match.group(1)}"
            
        return entities
    
    def _extract_cab_entities(self, text: str) -> Dict[str, Any]:
        entities = {}
        
        passenger_match = re.search(r'(\d+)\s*(passenger|people)', text.lower())
        if passenger_match:
            entities['passengers_count'] = int(passenger_match.group(1))
            
        return entities
    
    def perform_web_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Perform web search for non-standard requests"""
        try:
            with DDGS() as ddgs:
                results = []
                search_results = list(ddgs.text(query, max_results=max_results))
                
                for result in search_results:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", "")
                    })
                
                return results
        except Exception as e:
            st.error(f"Web search failed: {str(e)}")
            return [{"title": "Search Error", "url": "", "snippet": f"Unable to perform web search: {str(e)}"}]
    
    def process_request(self, user_input: str) -> Dict[str, Any]:
        """Main method to process user request using log probabilities"""
        
        # Step 1: Get category probabilities
        st.info("🧠 Calculating category probabilities...")
        category_log_probs = self.get_category_probabilities(user_input)
        
        # Step 2: Find the category with maximum log probability
        best_category = max(category_log_probs.keys(), key=lambda k: category_log_probs[k])
        max_log_prob = category_log_probs[best_category]
        
        # Step 3: Convert log probability to confidence score (0-1 range)
        # Using softmax normalization to convert log probs to probabilities
        log_prob_values = np.array(list(category_log_probs.values()))
        prob_values = np.exp(log_prob_values - np.max(log_prob_values))  # Numerical stability
        prob_values = prob_values / np.sum(prob_values)  # Normalize
        
        max_prob_index = np.argmax(log_prob_values)
        confidence_score = prob_values[max_prob_index]
        
        # Step 4: Extract entities and generate questions
        st.info(f"🎯 Detected category: {best_category} (confidence: {confidence_score:.1%})")
        entity_data = self.extract_entities_and_questions(user_input, best_category)
        
        # Step 5: Perform web search if category is "other"
        web_search_results = None
        if best_category == "other":
            st.info("🌐 Performing web search for additional information...")
            web_search_results = self.perform_web_search(user_input)
        
        # Convert log probabilities to regular probabilities for display
        display_probs = {}
        for i, category in enumerate(self.categories):
            display_probs[category] = float(prob_values[i])
        
        # Create response
        response = AssistantResponse(
            intent_category=best_category,
            key_entities=entity_data["key_entities"],
            confidence_score=float(confidence_score),
            category_probabilities=display_probs,
            follow_up_questions=entity_data["follow_up_questions"],
            web_search_results=web_search_results
        )
        
        return response.dict()

def main():
    st.set_page_config(
        page_title="Personal Assistant Bot with Log Probabilities",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Personal Assistant Bot with Log Probabilities")
    st.markdown("Convert fuzzy requests into structured responses using AI with explicit probability scoring!")
    
    # Initialize session state
    if 'google_api_key' not in st.session_state:
        st.session_state.google_api_key = ""
    if 'request_history' not in st.session_state:
        st.session_state.request_history = []
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "Google API Key", 
            type="password",
            value=st.session_state.google_api_key,
            help="Get your API key from Google AI Studio: https://makersuite.google.com/app/apikey"
        )
        if api_key:
            st.session_state.google_api_key = api_key
        
        st.markdown("---")
        st.markdown("### 💡 Example Requests:")
        
        examples = [
            "Need a sunset-view table for two tonight; gluten-free menu a must",
            "Planning a trip to Paris for 4 people next month",
            "Need a gift for my mom's birthday under $100",
            "Book a cab to airport tomorrow morning",
            "How to update Aadhar address online"
        ]
        
        for example in examples:
            if st.button(f"📝 {example[:30]}...", key=f"example_{hash(example)}", help=example):
                st.session_state.example_input = example
        
        st.markdown("---")
        st.markdown("### 📊 Supported Categories:")
        st.markdown("""
        - 🍽️ **Dining**: Restaurants, reservations
        - ✈️ **Travel**: Trips, flights, hotels  
        - 🎁 **Gifting**: Presents, occasions
        - 🚗 **Cab Booking**: Transportation
        - 🔍 **Other**: Web search for anything else
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Log Probability Features:")
        st.markdown("""
        - **Explicit Scoring**: Uses log probabilities for each category
        - **Maximum Selection**: Highest log-prob category is chosen
        - **Confidence Score**: Derived from probability distribution
        - **Transparent**: Shows probabilities for all categories
        """)
        
        # Request history section
        if st.session_state.request_history:
            st.markdown("---")
            st.markdown("### 📋 Recent Requests:")
            for i, req in enumerate(reversed(st.session_state.request_history[-5:]), 1):
                if st.button(f"{i}. {req[:25]}...", key=f"history_{i}", help=req):
                    st.session_state.example_input = req
    
    if not api_key:
        st.warning("⚠️ Please enter your Google API Key in the sidebar to continue.")
        st.info("💡 Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
        return
    
    # Initialize assistant
    try:
        assistant = PersonalAssistant(api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize assistant: {str(e)}")
        return
    
    # Input section
    st.header("📝 Enter Your Request")
    user_input = st.text_area(
        "What can I help you with?",
        value=st.session_state.get('example_input', ''),
        placeholder="How to update Aadhar address online",
        height=100,
        help="Type any request in natural language"
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        process_button = st.button("🚀 Process", type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear", help="Clear the input field")
    
    if clear_button:
        st.rerun()
    
    if process_button and user_input.strip():
        # Clear the example input after processing starts
        if 'example_input' in st.session_state:
            del st.session_state.example_input
            
        # Add to history
        if user_input not in st.session_state.request_history:
            st.session_state.request_history.append(user_input)
            if len(st.session_state.request_history) > 20:  # Keep only last 20
                st.session_state.request_history.pop(0)
        
        with st.spinner("🤖 Processing your request using log probabilities..."):
            try:
                result = assistant.process_request(user_input)
                
                # Display results with better formatting
                st.header("📊 Structured Response")
                
                # Intent and confidence in colored metrics
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    confidence = result['confidence_score']
                    confidence_color = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
                    st.metric(
                        "🎯 Intent Category", 
                        result['intent_category'].replace('_', ' ').title(),
                        help=f"AI detected this as a {result['intent_category']} request"
                    )
                
                with col2:
                    st.metric(
                        f"{confidence_color} Confidence", 
                        f"{confidence:.1%}",
                        help="Confidence derived from log probabilities"
                    )
                
                with col3:
                    entity_count = len(result['key_entities'])
                    st.metric(
                        "📋 Entities Found", 
                        entity_count,
                        help="Number of key pieces of information extracted"
                    )
                
                # Category probabilities visualization
                st.subheader("📈 Category Probabilities")
                prob_data = result['category_probabilities']
                
                # Create probability bars with better formatting
                for category, prob in prob_data.items():
                    is_selected = category == result['intent_category']
                    color = "🟢" if is_selected else "⚪"
                    bar_width = max(1, int(prob * 50))  # Ensure minimum width of 1
                    bar = "█" * bar_width + "░" * (50 - bar_width)
                    
                    # Format probability as percentage with appropriate precision
                    if prob >= 0.01:  # 1% or higher
                        prob_display = f"{prob:.1%}"
                    else:  # Less than 1%
                        prob_display = f"{prob:.2%}"
                    
                    st.write(f"{color} **{category.replace('_', ' ').title()}**: {prob_display}")
                    st.write(f"`{bar}`")
                    st.write("")
                
                # Key entities in an organized way
                if result['key_entities']:
                    st.subheader("🔍 Key Information Extracted")
                    
                    # Create columns for entities
                    entity_cols = st.columns(min(3, len(result['key_entities'])))
                    for i, (key, value) in enumerate(result['key_entities'].items()):
                        with entity_cols[i % 3]:
                            st.info(f"**{key.replace('_', ' ').title()}**\n\n{value}")
                
                # Follow-up questions
                if result['follow_up_questions']:
                    st.subheader("❓ Follow-up Questions")
                    for i, question in enumerate(result['follow_up_questions'], 1):
                        st.write(f"**{i}.** {question}")
                
                # Web search results with better formatting
                if result.get('web_search_results'):
                    st.subheader("🌐 Related Web Results")
                    st.info(f"Found {len(result['web_search_results'])} relevant results")
                    
                    for i, search_result in enumerate(result['web_search_results'], 1):
                        with st.expander(f"🔗 {search_result['title']}", expanded=i==1):
                            st.write(search_result['snippet'])
                            if search_result['url']:
                                st.markdown(f"**🌐 Source:** [{search_result['url']}]({search_result['url']})")
                
                # JSON output in an expandable section
                with st.expander("📄 Complete JSON Response", expanded=False):
                    st.code(json.dumps(result, indent=2), language='json')
                
                # Download options
                col1, col2 = st.columns([1, 1])
                with col1:
                    json_str = json.dumps(result, indent=2)
                    st.download_button(
                        label="💾 Download JSON",
                        data=json_str,
                        file_name=f"assistant_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Create a formatted text version
                    prob_text = "\n".join([f"- {k.replace('_', ' ').title()}: {v:.1%}" for k, v in result['category_probabilities'].items()])
                    
                    text_output = f"""Personal Assistant Response (Log Probabilities)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Input: {user_input}

Intent Category: {result['intent_category'].replace('_', ' ').title()}
Confidence Score: {result['confidence_score']:.1%}

Category Probabilities:
{prob_text}

Key Entities:
{chr(10).join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in result['key_entities'].items()])}

Follow-up Questions:
{chr(10).join([f"{i}. {q}" for i, q in enumerate(result['follow_up_questions'], 1)])}
"""
                    st.download_button(
                        label="📝 Download Text",
                        data=text_output,
                        file_name=f"assistant_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"❌ Error processing request: {str(e)}")
                st.info("💡 Try rephrasing your request or check your API key")
    
    elif process_button:
        st.warning("⚠️ Please enter a request to process.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Made with ❤️ using LangChain + Google Gemini AI + Log Probabilities<br>
        <small>This tool uses explicit log probabilities for more transparent intent classification</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
# main.py
import os
from typing import TypedDict, List, Optional, Literal
from datetime import datetime
import hashlib
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import requests
from typing_extensions import TypedDict
import re
from firecrawl import Firecrawl

load_dotenv()



# ==================== STATE DEFINITION ====================
class AgentState(TypedDict):
    """State for the document processing workflow"""
    url: str
    url_valid: Optional[bool]
    media_type: Optional[Literal["video", "text", "unsupported"]]
    raw_content: Optional[str]
    extracted_text: Optional[str]
    summary: Optional[str]
    code_snippets: Optional[List[str]]
    content_relevant: Optional[bool]
    content_type: Optional[str]  # e.g., "typescript", "python", "react", "sdk"
    markdown_content: Optional[str]

    error_message: Optional[str]
    metadata: dict

# ==================== NODE IMPLEMENTATIONS ====================
class URLValidator:
    """Node 1: Validate URL"""
    
    @staticmethod
    def validate_url(state: AgentState) -> AgentState:
        url = state.get("url", "")
        state["metadata"] = state.get("metadata", {})
        
        if not url:
            state["url_valid"] = False
            state["error_message"] = "No URL provided"
            return state
        
        # Check URL format
        url_pattern = re.compile(
            r'^(https?://)?'  # http:// or https://
            r'(www\.)?'  # optional www
            r'([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}'  # domain
            r'(:\d+)?'  # optional port
            r'(/.*)?$'  # path
        )
        
        if not url_pattern.match(url):
            state["url_valid"] = False
            state["error_message"] = f"Invalid URL format: {url}"
            return state
        
        # Try to fetch the URL
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
            
            if response.status_code >= 400:
                state["url_valid"] = False
                state["error_message"] = f"URL returned status code: {response.status_code}"
            else:
                state["url_valid"] = True
                state["metadata"]["content_type"] = response.headers.get('Content-Type', '')
                state["metadata"]["status_code"] = response.status_code
                
        except requests.RequestException as e:
            state["url_valid"] = False
            state["error_message"] = f"Failed to access URL: {str(e)}"
        
        return state

class MediaClassifier:
    """Node 2: Classify media type"""
    
    @staticmethod
    def classify_media(state: AgentState) -> AgentState:
        if not state.get("url_valid", False):
            state["media_type"] = "unsupported"
            return state
        
        url = state["url"].lower()
        content_type = state["metadata"].get("content_type", "").lower()
        
        # Check for video platforms
        video_domains = ["youtube.com", "youtu.be", "vimeo.com", "dailymotion.com"]
        if any(domain in url for domain in video_domains):
            state["media_type"] = "video"
        # Check content type
        elif "video" in content_type:
            state["media_type"] = "video"
        elif "text/html" in content_type or "application/pdf" in content_type:
            state["media_type"] = "text"
        else:
            # Fallback: check URL pattern
            if any(ext in url for ext in ['.pdf', '.html', '.htm', '.md', '.txt']):
                state["media_type"] = "text"
            else:
                state["media_type"] = "text"  # Default to text for webpages
        
        return state

class ContentProcessor:
    """Unified node that combines Scraper, Cleaner, and Adder functionality"""
    
    def __init__(self, firecrawl_api_key: str = None, openai_api_key: str = None):
        self.firecrawl_api_key = firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")
        if not self.firecrawl_api_key:
            raise ValueError("FIRECRAWL_API_KEY environment variable is required")
        
        self.llm = ChatOpenAI(
            model="gpt-5-nano",
            temperature=0,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
    
    def process_content(self, state: AgentState) -> AgentState:
        """Unified method that scrapes, cleans, and enhances content in one go"""
        url = state["url"]
        media_type = state.get("media_type", "text")
        
        print(f"DEBUG ContentProcessor - Starting processing for URL: {url}")
        print(f"DEBUG ContentProcessor - media_type: {media_type}")
        
        # Step 1: Scrape content
        try:
            if media_type == "video":
                state["error_message"] = "Video content should be handled separately. Use text URLs for Firecrawl scraping."
                return state
            
            app = Firecrawl(api_key=self.firecrawl_api_key)
            scrape_result = app.scrape(url, formats=['markdown'])
            
            if not scrape_result or not hasattr(scrape_result, 'markdown') or not scrape_result.markdown:
                state["error_message"] = f"No markdown content found at URL: {url}. Firecrawl response: {scrape_result}"
                return state
            
            markdown_content = scrape_result.markdown
            
            if not markdown_content or not markdown_content.strip():
                state["error_message"] = "Firecrawl returned empty markdown content"
                return state
            
            state["raw_content"] = markdown_content
            state["extracted_text"] = markdown_content
            state["metadata"]["scrape_method"] = "firecrawl_sync_markdown"
            state["metadata"]["content_length"] = len(markdown_content)
            
            # Extract code snippets from markdown
            code_snippets = []
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', markdown_content, re.DOTALL)
            code_snippets.extend(code_blocks)
            
            inline_code = re.findall(r'`([^`]+)`', markdown_content)
            code_snippets.extend([code for code in inline_code if len(code) > 3])
            
            state["code_snippets"] = code_snippets
            
            print(f"DEBUG ContentProcessor - scraped content length: {len(markdown_content)}")
            
        except Exception as e:
            state["error_message"] = f"Firecrawl scraping failed: {str(e)}"
            return state
        
        # Step 2: Clean content
        content = state.get("extracted_text", "")
        
        if content and content.strip():
            try:
                clean_prompt = ChatPromptTemplate.from_template("""
                You are cleaning up markdown content scraped from a documentation website. 
                Remove all unnecessary parts while preserving the valuable technical content.
                
                Content to clean:
                {content}
                
                REMOVE these unnecessary parts:
                - Navigation menus, headers, footers, sidebars
                - Advertisements and promotional content
                - "Learn more", "Read more", "See also" links
                - Social media sharing buttons, cookie notices
                - Author bios, publication dates (unless relevant)
                - Repeated or redundant information
                - Excessive whitespace and poor formatting
                - Non-technical filler text and fluff
                
                KEEP these valuable parts:
                - Technical concepts and explanations
                - Code examples and code blocks
                - Step-by-step instructions
                - Best practices and guidelines
                - Configuration instructions
                - API references and method signatures
                - Diagrams, tables, and structured data
                - Important warnings or notes
                
                Return ONLY the cleaned markdown content. Do not add explanations or headers.
                Ensure all code blocks remain properly formatted with ```language syntax.
                """)
                
                clean_chain = clean_prompt | self.llm | StrOutputParser()
                cleaned_content = clean_chain.invoke({"content": content[:12000]})
                
                if cleaned_content and cleaned_content.strip():
                    state["extracted_text"] = cleaned_content
                    state["metadata"]["content_cleaned"] = True
                    state["metadata"]["original_length"] = len(content)
                    state["metadata"]["cleaned_length"] = len(cleaned_content)
                    print(f"DEBUG ContentProcessor - cleaned content length: {len(cleaned_content)}")
                else:
                    state["metadata"]["content_cleaned"] = False
                    state["metadata"]["clean_status"] = "Cleaning failed, preserving original"
                    if content:
                        state["extracted_text"] = content
                
            except Exception as e:
                state["metadata"]["content_cleaned"] = False
                state["metadata"]["clean_status"] = f"Cleaning failed: {str(e)}, preserving original"
        
        # Step 3: Enhance content
        enhanced_content = state.get("extracted_text", "")
        
        if enhanced_content and enhanced_content.strip():
            try:
                enhance_prompt = ChatPromptTemplate.from_template("""
                Enhance this technical content by adding best practices and "DO vs DON'T" guidelines.
                
                Content to enhance:
                {content}
                
                Source: {url}
                Media Type: {media_type}
                
                Add value by:
                1. Identifying key practices and adding "✅ DO" recommendations
                2. Highlighting common mistakes and adding "❌ DON'T" warnings
                3. Adding specific use case considerations
                4. Including performance tips or security considerations
                5. Providing alternative approaches when relevant
                
                Format the enhanced content with:
                - Original content preserved
                - Clear DO's and DON'Ts sections
                - Practical examples for each guideline
                - Specific context for different use cases
                
                Return the enhanced markdown content:
                """)
                
                enhance_chain = enhance_prompt | self.llm | StrOutputParser()
                final_content = enhance_chain.invoke({
                    "content": enhanced_content,
                    "url": url,
                    "media_type": media_type
                })
                
                state["extracted_text"] = final_content
                state["markdown_content"] = final_content
                state["metadata"]["content_enhanced"] = True
                print(f"DEBUG ContentProcessor - enhanced content length: {len(final_content)}")
                
            except Exception as e:
                state["error_message"] = f"Content enhancement failed: {str(e)}"
        
        return state





# ==================== GRAPH CONSTRUCTION ====================
def create_workflow(openai_api_key: str = None, firecrawl_api_key: str = None):
    """Create the LangGraph workflow with unified ContentProcessor"""
    
    # Initialize nodes
    validator = URLValidator()
    classifier = MediaClassifier()
    content_processor = ContentProcessor(firecrawl_api_key, openai_api_key)
    
    # Define node functions
    def validate_url_node(state: AgentState):
        return validator.validate_url(state)
    
    def classify_media_node(state: AgentState):
        return classifier.classify_media(state)
    
    def content_processor_node(state: AgentState):
        return content_processor.process_content(state)
    

    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("validate_url", validate_url_node)
    workflow.add_node("classify_media", classify_media_node)
    workflow.add_node("content_processor", content_processor_node)
    
    # Add edges (simplified pipeline: ContentProcessor → Save)
    workflow.set_entry_point("validate_url")
    
    # From validate_url to classify_media
    workflow.add_edge("validate_url", "classify_media")
    
    # From classify_media to content_processor (Router - both video and text go to content_processor)
    def route_to_content_processor(state: AgentState):
        if state.get("url_valid", False) is False:
            return END
        media_type = state.get("media_type")
        if media_type in ["video", "text"]:
            return "content_processor"
        else:
            return END
    
    workflow.add_conditional_edges(
        "classify_media",
        route_to_content_processor,
        {
            "content_processor": "content_processor",
            END: END
        }
    )
    
    # From content_processor to END (workflow ends here)
    workflow.add_edge("content_processor", END)
    
    # Compile the graph
    return workflow.compile()

# ==================== MAIN EXECUTION ====================
def main():
    """Main function to run the workflow"""
    # Set your OpenAI API key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = input("Enter your OpenAI API key: ").strip()
    
    # Set your Firecrawl API key
    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
    
    if not FIRECRAWL_API_KEY:
        FIRECRAWL_API_KEY = input("Enter your Firecrawl API key: ").strip()
    
    # Create the workflow
    app = create_workflow(OPENAI_API_KEY, FIRECRAWL_API_KEY)
    
    url = input("URL: ").strip()
    
    if not url:
        return
    
    # Initial state
    initial_state = AgentState(
        url=url,
        url_valid=None,
        media_type=None,
        raw_content=None,
        extracted_text=None,
        summary=None,
        code_snippets=[],
        content_relevant=None,
        content_type=None,
        markdown_content=None,

        error_message=None,
        metadata={}
    )
    
    try:
        final_state = app.invoke(initial_state)
        
        if final_state.get("error_message"):
            print(f"Error: {final_state['error_message']}")
            return
        else:
            print(f"Success! Content processed. Markdown content length: {len(final_state.get('markdown_content', ''))}")
    
    except Exception as e:
        print(f"Workflow failed: {str(e)}")

if __name__ == "__main__":
    main()
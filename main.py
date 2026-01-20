# main.py
import os
from typing import TypedDict, List, Optional, Literal
from datetime import datetime
import hashlib
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import requests
from typing_extensions import TypedDict
import re

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
    file_path: Optional[str]
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

class VideoProcessor:
    """Node 3a: Process video content"""
    
    def __init__(self, openai_api_key: str = None):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
    
    def process_video(self, state: AgentState) -> AgentState:
        url = state["url"]
        
        try:
            # Try YouTube first
            if "youtube.com" in url or "youtu.be" in url:
                # Convert youtu.be short URL to full youtube.com URL for better compatibility
                video_id = url.split("/")[-1].split("?")[0]
                full_url = f"https://www.youtube.com/watch?v={video_id}"
                loader = YoutubeLoader.from_youtube_url(
                    full_url, 
                    add_video_info=False,
                    language=['en']
                )
            else:
                loader = YoutubeLoader.from_youtube_url(
                    url, 
                    add_video_info=False,
                    language=['en']
                )
            
            documents = loader.load()
            if documents:
                state["raw_content"] = documents[0].page_content
                state["metadata"]["video_title"] = documents[0].metadata.get('title', 'Unknown Title')
                state["metadata"]["author"] = documents[0].metadata.get('author', 'Unknown Author')
            else:
                state["error_message"] = f"No transcript available for this video and Unsupported video platform: {url}"
                return state
            
            # Summarize video content
            prompt = ChatPromptTemplate.from_template("""
            Summarize the following video transcript concisely, focusing on coding best practices, 
            technical concepts, and important instructions. Extract any code examples or patterns mentioned.
            
            Transcript:
            {transcript}
            
            Provide a structured summary with:
            1. Main topic and key takeaways
            2. Code examples and patterns
            3. Best practices mentioned
            4. Tools/libraries/frameworks discussed
            
            Summary:
            """)
            
            chain = prompt | self.llm | StrOutputParser()
            summary = chain.invoke({"transcript": state["raw_content"][:4000]})
            state["summary"] = summary
            
            # Extract text from summary
            state["extracted_text"] = summary
            
        except Exception as e:
            state["error_message"] = f"Video processing failed: {str(e)}"
        
        return state

class TextProcessor:
    """Node 3b: Process text/code content"""
    
    def __init__(self, openai_api_key: str = None):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
    
    def process_text(self, state: AgentState) -> AgentState:
        url = state["url"]
        
        try:
            # Load web content
            loader = WebBaseLoader([url])
            documents = loader.load()
            
            if not documents:
                state["error_message"] = "No content found at URL"
                return state
            
            full_text = documents[0].page_content
            state["raw_content"] = full_text
            
            # Extract important text and code snippets
            prompt = ChatPromptTemplate.from_template("""
            Extract the most important information from this documentation/content.
            Focus on:
            1. Key concepts and explanations
            2. Code snippets and examples
            3. Best practices and guidelines
            4. Configuration instructions
            5. API references or method signatures
            
            Content:
            {content}
            
            Provide the extracted information in this format:
            IMPORTANT_TEXT:
            [concise extraction of key textual information]
            
            CODE_SNIPPETS:
            [list of code snippets found, each marked with language if specified]
            
            BEST_PRACTICES:
            [list of best practices mentioned]
            """)
            
            chain = prompt | self.llm | StrOutputParser()
            extraction = chain.invoke({"content": full_text[:5000]})
            
            # Parse the extraction
            sections = extraction.split("\n\n")
            state["extracted_text"] = extraction
            
            # Extract code snippets
            code_snippets = []
            for section in sections:
                if "```" in section:
                    # Extract code between backticks
                    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', section, re.DOTALL)
                    code_snippets.extend(code_blocks)
            
            state["code_snippets"] = code_snippets
            
        except Exception as e:
            state["error_message"] = f"Text processing failed: {str(e)}"
        
        return state

class QualityChecker:
    """Node 4: Check content quality and relevance"""
    
    def __init__(self, openai_api_key: str = None):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
    
    def check_quality(self, state: AgentState) -> AgentState:
        content = state.get("extracted_text", "") or state.get("summary", "")
        
        if not content:
            state["content_relevant"] = False
            state["error_message"] = "No content to check"
            return state
        
        prompt = ChatPromptTemplate.from_template("""
        Analyze if this content is relevant to coding/development best practices.
        
        Consider:
        1. Is it about programming, software development, or coding practices?
        2. Does it contain technical information useful for developers?
        3. Is it misleading or spam?
        4. Does it contain harmful or inappropriate content?
        
        Content:
        {content}
        
        Respond ONLY with JSON:
        {{
            "relevant": boolean,
            "reason": "brief explanation",
            "primary_topic": "main topic detected"
        }}
        """)
        
        try:
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"content": content[:2000]})
            
            # Parse JSON response
            import json
            result = json.loads(response)
            
            state["content_relevant"] = result.get("relevant", False)
            state["metadata"]["relevance_reason"] = result.get("reason", "")
            state["metadata"]["primary_topic"] = result.get("primary_topic", "")
            
        except Exception as e:
            state["content_relevant"] = False
            state["error_message"] = f"Quality check failed: {str(e)}"
        
        return state

class TemplateDetector:
    """Node 5: Detect content type for template selection"""
    
    def __init__(self, openai_api_key: str = None):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
    
    def detect_template(self, state: AgentState) -> AgentState:
        content = state.get("extracted_text", "") or state.get("summary", "")
        
        if not content:
            return state
        
        prompt = ChatPromptTemplate.from_template("""
        Based on this technical content, determine what type of coding/documentation this is.
        
        Categories:
        - programming_language: (python, javascript, typescript, java, go, rust, cpp, csharp, etc.)
        - framework: (react, vue, angular, django, flask, spring, .net, etc.)
        - library: (numpy, pandas, tensorflow, pytorch, etc.)
        - sdk: (aws-sdk, azure-sdk, google-cloud-sdk, etc.)
        - tool: (docker, kubernetes, git, ci-cd, etc.)
        - general: (general best practices, software architecture, design patterns)
        
        Content snippet:
        {content}
        
        Respond ONLY with the primary category from the list above.
        If uncertain, respond with "general".
        """)
        
        try:
            chain = prompt | self.llm | StrOutputParser()
            content_type = chain.invoke({"content": content[:1500]})
            
            # Clean up response
            content_type = content_type.strip().lower()
            state["content_type"] = content_type
            
        except Exception as e:
            state["content_type"] = "general"
            state["error_message"] = f"Template detection failed: {str(e)}"
        
        return state

class TemplateCreator:
    """Node 6: Create markdown template"""
    
    def __init__(self, openai_api_key: str = None):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
    
    def create_markdown(self, state: AgentState) -> AgentState:
        if not state.get("content_relevant", False):
            state["error_message"] = "Content not relevant, skipping markdown creation"
            return state
        
        content = state.get("extracted_text", "") or state.get("summary", "")
        content_type = state.get("content_type", "general")
        url = state["url"]
        code_snippets = state.get("code_snippets", [])
        
        prompt = ChatPromptTemplate.from_template("""
        Create a comprehensive, detailed markdown document for a coding agent based on the following content.
        
        Content Type: {content_type}
        Source URL: {url}
        
        Content:
        {content}
        
        {code_snippets_section}
        
        Create a markdown document with these sections:
        # [Detailed Title based on content]
        
        ## Source Information
        - **URL**: {url}
        - **Content Type**: {content_type}
        - **Generated**: {timestamp}
        - **Last Updated**: {timestamp}
        
        ## Executive Summary
        - Brief overview of the main concepts
        - Key takeaways for developers
        - Prerequisites and requirements
        - Target audience level
        
        ## Core Concepts & Fundamentals
        - Detailed explanation of main concepts
        - Theoretical background where applicable
        - Key terminology and definitions
        - How concepts relate to each other
        - Real-world use cases and scenarios
        
        ## Detailed Implementation Guide
        - Step-by-step implementation process
        - Configuration requirements
        - Environment setup instructions
        - Dependencies and version requirements
        - Installation and setup procedures
        
        ## Comprehensive Best Practices & Guidelines
        - Performance optimization techniques
        - Security considerations and measures
        - Error handling strategies
        - Code organization and structure
        - Testing and validation approaches
        - Documentation standards
        
        ## Complete Code Examples & Patterns
        {code_examples_placeholder}
        - Production-ready examples
        - Common design patterns
        - Integration examples with other tools/frameworks
        - Full working implementations
        - Code walkthroughs and explanations
        
        ## DO's and DON'Ts - Critical Guidelines
        ### ✅ DO's
        - List of recommended practices and approaches
        - Things you should always do
        - Best practices to follow
        - Recommended configurations and settings
        - Proper error handling approaches
        
        ### ❌ DON'Ts
        - Common mistakes to avoid
        - Anti-patterns and problematic approaches
        - Things you should never do
        - Deprecated methods or practices
        - Security vulnerabilities to avoid
        
        ## Advanced Techniques & Optimizations
        - Performance tuning strategies
        - Advanced usage patterns
        - Scaling considerations
        - Optimization tips and tricks
        - Edge cases and how to handle them
        
        ## Troubleshooting & Debugging
        - Common issues and their solutions
        - Debugging techniques and tools
        - Error message meanings and resolutions
        - Performance bottlenecks and how to identify them
        - Logging and monitoring recommendations
        
        ## Integration & Compatibility
        - Integration with popular frameworks and libraries
        - API compatibility considerations
        - Database integration patterns
        - Third-party service integrations
        - Version compatibility matrix
        
        ## Testing Strategies
        - Unit testing approaches and frameworks
        - Integration testing strategies
        - Performance testing methodologies
        - Test-driven development (TDD) practices
        - Continuous integration considerations
        
        ## Security Considerations
        - Common security vulnerabilities
        - Authentication and authorization patterns
        - Data protection and encryption
        - Secure coding practices
        - Compliance and regulatory considerations
        
        ## Performance Metrics & Monitoring
        - Key performance indicators (KPIs)
        - Monitoring tools and techniques
        - Performance benchmarking
        - Resource utilization optimization
        - Analytics and reporting
        
        ## Common Pitfalls & Solutions
        - Frequently encountered problems
        - Root cause analysis of common issues
        - Preventive measures
        - Recovery strategies
        - Lessons learned from real-world deployments
        
        ## Frequently Asked Questions (FAQ)
        - Common developer questions
        - Clarification of confusing concepts
        - Workarounds for known limitations
        - Alternative approaches when primary methods fail
        
        ## Additional Resources & References
        - Official documentation links
        - Recommended books and tutorials
        - Community forums and discussion groups
        - Related tools and libraries
        - Further reading and learning resources
        - Video tutorials and courses
        
        ## Quick Reference Cheat Sheet
        - Common commands and syntax
        - Quick configuration snippets
        - Frequently used code patterns
        - Essential parameters and their meanings
        
        Make the content extremely detailed, practical, and actionable.
        Include real-world examples, use cases, and scenario-based explanations.
        Format all code blocks with proper language specification.
        Ensure the content is comprehensive enough to serve as a complete reference guide.
        """)
        
        try:
            # Prepare code snippets section
            code_snippets_text = ""
            if code_snippets:
                code_snippets_text = "\n".join([f"```\n{snippet[:500]}\n```" for snippet in code_snippets[:3]])
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            chain = prompt | self.llm | StrOutputParser()
            markdown_content = chain.invoke({
                "content": content[:6000],
                "content_type": content_type,
                "url": url,
                "timestamp": timestamp,
                "code_snippets_section": f"Code Snippets:\n{code_snippets_text}" if code_snippets_text else "",
                "code_examples_placeholder": "[Include relevant code examples here]" if not code_snippets else ""
            })
            
            state["markdown_content"] = markdown_content
            
        except Exception as e:
            state["error_message"] = f"Markdown creation failed: {str(e)}"
        
        return state

class TemplateSaver:
    """Node 7: Save markdown file locally"""
    
    @staticmethod
    def save_template(state: AgentState) -> AgentState:
        if not state.get("markdown_content"):
            state["error_message"] = "No markdown content to save"
            return state
        
        try:
            # Create a filename from URL and timestamp
            url_hash = hashlib.md5(state["url"].encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            content_type = state.get("content_type", "unknown").replace("-", "_").replace(" ", "_")
            
            filename = f"best_practices_{content_type}_{url_hash}_{timestamp}.md"
            
            # Save to downloads folder or desktop
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            
            # Try downloads first, then desktop
            save_path = None
            if os.path.exists(downloads_path):
                save_path = os.path.join(downloads_path, filename)
            elif os.path.exists(desktop_path):
                save_path = os.path.join(desktop_path, filename)
            else:
                # Fallback to current directory
                save_path = filename
            
            # Save the file
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(state["markdown_content"])
            
            state["file_path"] = save_path
            
        except Exception as e:
            state["error_message"] = f"Failed to save file: {str(e)}"
        
        return state

# ==================== GRAPH CONSTRUCTION ====================
def create_workflow(openai_api_key: str = None):
    """Create the LangGraph workflow"""
    
    # Initialize nodes
    validator = URLValidator()
    classifier = MediaClassifier()
    video_processor = VideoProcessor(openai_api_key)
    text_processor = TextProcessor(openai_api_key)
    quality_checker = QualityChecker(openai_api_key)
    template_detector = TemplateDetector(openai_api_key)
    template_creator = TemplateCreator(openai_api_key)
    template_saver = TemplateSaver()
    
    # Define node functions
    def validate_url_node(state: AgentState):
        return validator.validate_url(state)
    
    def classify_media_node(state: AgentState):
        return classifier.classify_media(state)
    
    def process_video_node(state: AgentState):
        return video_processor.process_video(state)
    
    def process_text_node(state: AgentState):
        return text_processor.process_text(state)
    
    def check_quality_node(state: AgentState):
        return quality_checker.check_quality(state)
    
    def detect_template_node(state: AgentState):
        return template_detector.detect_template(state)
    
    def create_markdown_node(state: AgentState):
        return template_creator.create_markdown(state)
    
    def save_template_node(state: AgentState):
        return template_saver.save_template(state)
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("validate_url", validate_url_node)
    workflow.add_node("classify_media", classify_media_node)
    workflow.add_node("process_video", process_video_node)
    workflow.add_node("process_text", process_text_node)
    workflow.add_node("check_quality", check_quality_node)
    workflow.add_node("detect_template", detect_template_node)
    workflow.add_node("create_markdown", create_markdown_node)
    workflow.add_node("save_template", save_template_node)
    
    # Add edges (conditional routing)
    workflow.set_entry_point("validate_url")
    
    # From validate_url to classify_media
    workflow.add_edge("validate_url", "classify_media")
    
    # From classify_media to appropriate processor
    def route_media(state: AgentState):
        if state.get("url_valid", False) is False:
            return END
        media_type = state.get("media_type")
        if media_type == "video":
            return "process_video"
        elif media_type == "text":
            return "process_text"
        else:
            return END
    
    workflow.add_conditional_edges(
        "classify_media",
        route_media,
        {
            "process_video": "process_video",
            "process_text": "process_text",
            END: END
        }
    )
    
    # From processors to quality check
    workflow.add_edge("process_video", "check_quality")
    workflow.add_edge("process_text", "check_quality")
    
    # From quality check to template detection (if relevant)
    def after_quality_check(state: AgentState):
        if state.get("content_relevant", False):
            return "detect_template"
        else:
            return END
    
    workflow.add_conditional_edges(
        "check_quality",
        after_quality_check,
        {
            "detect_template": "detect_template",
            END: END
        }
    )
    
    # From template detection to markdown creation
    workflow.add_edge("detect_template", "create_markdown")
    
    # From markdown creation to saving
    def after_markdown_creation(state: AgentState):
        if state.get("markdown_content"):
            return "save_template"
        else:
            return END
    
    workflow.add_conditional_edges(
        "create_markdown",
        after_markdown_creation,
        {
            "save_template": "save_template",
            END: END
        }
    )
    
    # From saving to end
    workflow.add_edge("save_template", END)
    
    # Compile the graph
    return workflow.compile()

# ==================== MAIN EXECUTION ====================
def main():
    """Main function to run the workflow"""
    # Set your OpenAI API key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = input("Enter your OpenAI API key: ").strip()
    
    # Create the workflow
    app = create_workflow(OPENAI_API_KEY)
    
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
        file_path=None,
        error_message=None,
        metadata={}
    )
    
    try:
        final_state = app.invoke(initial_state)
        
        if final_state.get("error_message"):
            return
    
    except Exception as e:
        pass

if __name__ == "__main__":
    main()
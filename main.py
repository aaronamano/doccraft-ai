from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from agent import AgentState
import os
import re
from datetime import datetime

load_dotenv()

app = FastAPI(title="Code Context Crafter API")

templates = Jinja2Templates(directory=".")

class AgentRequest(BaseModel):
    url: str

@app.post("/agent")
async def process_url(request: AgentRequest):
    try:
        import os
        # Create workflow
        from agent import create_workflow
        openai_api_key = os.getenv("OPENAI_API_KEY")
        firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
        
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY in environment")
        if not firecrawl_api_key:
            raise HTTPException(status_code=500, detail="Missing FIRECRAWL_API_KEY in environment")
            
        workflow = create_workflow(openai_api_key, firecrawl_api_key)
        
        # Initial state
        initial_state = AgentState(
            url=request.url,
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
        
        # Run the workflow
        final_state = workflow.invoke(initial_state)
        
        # Check for errors
        if final_state.get("error_message"):
            raise HTTPException(status_code=400, detail=final_state.get("error_message"))
        
        # Create downloads directory if it doesn't exist
        os.makedirs("downloads", exist_ok=True)
        
        # Generate filename from URL
        safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', request.url)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_filename}_{timestamp}.md"
        filepath = os.path.join("downloads", filename)
        
        # Save markdown content to file
        markdown_content = final_state.get("markdown_content", "")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        return {
            "markdown_content": markdown_content,
            "file_path": filepath
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    filepath = os.path.join("downloads", filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(filepath, media_type='text/markdown', filename=filename)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
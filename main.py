from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from agent import AgentState

load_dotenv()

app = FastAPI(title="Code Context Crafter API")

templates = Jinja2Templates(directory=".")

class AgentRequest(BaseModel):
    url: str

@app.post("/agent")
async def process_url(request: AgentRequest):
    try:
        # Create the workflow
        from agent import create_workflow
        app = create_workflow()
        
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
            file_path=None,
            error_message=None,
            metadata={}
        )
        
        # Run the workflow
        final_state = app.invoke(initial_state)
        
        # Check for errors
        if final_state.get("error_message"):
            raise HTTPException(status_code=400, detail=final_state.get("error_message"))
        
        # Return only markdown content
        return final_state.get("markdown_content", "")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
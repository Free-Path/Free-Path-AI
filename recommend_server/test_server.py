from fastapi import FastAPI
import uvicorn
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Hello from FastAPI"}

@app.get("/test")
async def test():
    logger.info("Test endpoint accessed")
    return {
        "internal_ip": "10.100.39.196",
        "external_ip": "114.110.132.5",
        "status": "running"
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
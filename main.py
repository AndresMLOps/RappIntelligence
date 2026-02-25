# main.py
import uvicorn

def main():
    uvicorn.run(
        "scr.api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,         # recarga automática al guardar código
    )

if __name__ == "__main__":
    main()

"""ASGI entrypoint: uvicorn doc_agent.main:app"""

from doc_agent.api.app import create_app

app = create_app()

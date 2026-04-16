import os
from flask import Flask
from flask_cors import CORS
from api.routes import api_bp
from core.runner import RunQueue
from core.db import init_db


def create_app():
    app = Flask(__name__)

    CORS(app, origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Create React App fallback
    ])

    init_db(app)

    runner = RunQueue()
    runner.start()
    app.config["RUNNER"] = runner

    app.register_blueprint(api_bp, url_prefix="/api")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


if __name__ == "__main__":
    app = create_app()
    # debug=False is important — reloader would spawn a second runner thread
    app.run(host="0.0.0.0", port=5001, debug=False)
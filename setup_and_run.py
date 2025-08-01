#!/usr/bin/env python3
"""
Setup script to prepare the environment and run the FastAPI server
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n📋 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def setup_environment():
    """Setup the development environment"""
    print("🚀 Setting up Q.4 Retrieval System API")
    
    # Create necessary directories
    directories = ["pdfs", "chroma_db"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install requirements. Please check your Python environment.")
        return False
    
    # Check if Ollama is running (optional)
    print("\n🔍 Checking Ollama installation...")
    ollama_check = subprocess.run("ollama --version", shell=True, capture_output=True)
    if ollama_check.returncode == 0:
        print("✅ Ollama is installed")
        
        # Check if required models are available
        models_to_check = ["mistral", "nomic-embed-text"]
        for model in models_to_check:
            print(f"Checking model: {model}")
            model_check = subprocess.run(f"ollama list | grep {model}", shell=True, capture_output=True)
            if model_check.returncode != 0:
                print(f"⚠️  Model {model} not found. You may need to pull it:")
                print(f"   ollama pull {model}")
    else:
        print("⚠️  Ollama not found. Please install Ollama and required models:")
        print("   - Install Ollama from https://ollama.ai")
        print("   - Run: ollama pull mistral")
        print("   - Run: ollama pull nomic-embed-text")
    
    return True

def run_server():
    """Run the FastAPI server"""
    print("\n🌐 Starting FastAPI server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run("uvicorn main:app --host 0.0.0.0 --port 8000 --reload", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error running server: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run_server()
    else:
        if setup_environment():
            print("\n✅ Setup complete!")
            print("\nTo start the server, run:")
            print("   python setup_and_run.py run")
            print("Or directly:")
            print("   uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
        else:
            print("❌ Setup failed. Please check the errors above.")
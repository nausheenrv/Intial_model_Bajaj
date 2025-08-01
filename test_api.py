#!/usr/bin/env python3
"""
Test script for the Q.4 Retrieval System API
"""
import requests
import json

# API Configuration
BASE_URL = "http://localhost:8000"
TOKEN = "7c49d0c1af87904647ed2d5803a1f9678d7960387ad9c10ecb72e9ef27456e2b"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_api_endpoint():
    """Test the main API endpoint with sample data"""
    print("\n🧪 Testing /hackrx/run endpoint...")
    
    # Sample request data matching the format from the images
    test_data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?"
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {TOKEN}"
    }
    
    try:
        print("Sending request...")
        print(f"URL: {BASE_URL}/hackrx/run")
        print(f"Headers: {headers}")
        print(f"Data: {json.dumps(test_data, indent=2)}")
        
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            json=test_data,
            headers=headers,
            timeout=300  # 5 minutes timeout for processing
        )
        
        print(f"\nResponse Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (this is normal for first run as models need to load)")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_authentication():
    """Test authentication with invalid token"""
    print("\n🔐 Testing authentication...")
    
    test_data = {
        "documents": "https://example.com/test.pdf",
        "questions": ["Test question?"]
    }
    
    # Test with invalid token
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer invalid_token"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/hackrx/run", json=test_data, headers=headers)
        if response.status_code == 401:
            print("✅ Authentication working correctly (rejected invalid token)")
            return True
        else:
            print(f"❌ Authentication issue: got status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Q.4 Retrieval System API")
    print("=" * 50)
    
    # Run tests
    health_ok = test_health_check()
    auth_ok = test_authentication()
    
    if health_ok and auth_ok:
        print("\n🚀 Running main API test...")
        api_ok = test_api_endpoint()
        
        if api_ok:
            print("\n✅ All tests passed!")
        else:
            print("\n⚠️  Main API test failed, but basic functionality is working")
    else:
        print("\n❌ Basic tests failed. Check if the server is running.")
    
    print("\nTo start the server if not running:")
    print("   python setup_and_run.py run")

import requests
import time
import sys

def test_api():
    BASE_URL = "http://127.0.0.1:8000"
    
    # Wait for server
    print("Waiting for server...")
    for i in range(10):
        try:
            resp = requests.get(f"{BASE_URL}/health")
            if resp.status_code == 200:
                print("Server is up!")
                break
        except:
            time.sleep(1)
    else:
        print("Server failed to start.")
        sys.exit(1)

    # Test Prediction
    payload = {
        "attendance_percentage": 85,
        "assignment_average": 70,
        "internal_marks": 65,
        "previous_sem_gpa": 7.5
    }
    
    print(f"Testing /predict with {payload}")
    try:
        resp = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
        if resp.status_code == 200:
             print("Prediction test PASSED")
        else:
             print("Prediction test FAILED")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api()

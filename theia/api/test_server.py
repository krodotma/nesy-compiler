"""
Tests for Theia API Server.

Verifies:
- App creation
- Status endpoint
- Ingest endpoint
"""

import unittest
from fastapi.testclient import TestClient
from theia.api.server import create_app

class TestTheiaAPI(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = TestClient(cls.app)
    
    def test_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"service": "theia", "status": "active"})
        
    def test_status(self):
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["active"])
        self.assertIn("uptime", data)

    def test_ingest(self):
        payload = {"frames": ["base64fakeimage"], "source": "test"}
        response = self.client.post("/v1/theia/ingest", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 1)

if __name__ == "__main__":
    unittest.main()

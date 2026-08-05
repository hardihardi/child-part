import unittest
import os
import sqlite3
from database import init_db, get_db_connection, get_config
from app import app

class TestChildPartApp(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        init_db()

    def test_database_init(self):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Users")
        count = cursor.fetchone()[0]
        conn.close()
        self.assertGreaterEqual(count, 3)

    def test_get_config(self):
        config = get_config()
        self.assertIn('target_klip_lh', config)
        self.assertIn('target_klip_rh', config)

    def test_login_page_renders(self):
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()

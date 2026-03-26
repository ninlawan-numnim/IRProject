import unittest
from app import app, db, User, Folder, Bookmark


class FoodAppTestCase(unittest.TestCase):

    # ฟังก์ชันนี้จะทำงาน 'ก่อน' เริ่มรันแต่ละเทสต์เคส
    def setUp(self):
        # เปลี่ยนไปใช้ Database สำรองสำหรับเทสต์โดยเฉพาะ (จะได้ไม่กวนข้อมูลจริง)
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test_food_app.db'
        app.config['WTF_CSRF_ENABLED'] = False  # ปิด CSRF ชั่วคราวเพื่อให้เทสต์ฟอร์มได้ง่าย

        self.client = app.test_client()

        with app.app_context():
            db.create_all()
            # สร้าง User จำลองสำหรับใช้เทสต์
            user = User(username='testuser', password_hash='pbkdf2:sha256:250000$dummyhash')
            db.session.add(user)
            db.session.commit()

    # ฟังก์ชันนี้จะทำงาน 'หลัง' รันแต่ละเทสต์เคสเสร็จ
    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    # ==========================================
    # 1. Unit Test: ทดสอบระบบ Authentication
    # ==========================================
    def test_register_page_loads(self):
        """ตรวจสอบว่าหน้า Register โหลดได้ปกติ (Status Code 200)"""
        response = self.client.get('/register')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Register', response.data)

    def test_login_page_loads(self):
        """ตรวจสอบว่าหน้า Login โหลดได้ปกติ"""
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)

    # ==========================================
    # 2. Higher-level Test: ทดสอบระบบ Search & IR Features
    # ==========================================
    def test_search_requires_login(self):
        """ตรวจสอบว่าถ้ายังไม่ล็อกอิน จะเข้าหน้า Search ไม่ได้ (ต้องโดน Redirect)"""
        response = self.client.get('/search?q=chicken')
        self.assertEqual(response.status_code, 302)  # 302 คือการ Redirect กลับไปหน้า Login

    def test_autocomplete_api(self):
        """ตรวจสอบว่า API Autocomplete ทำงานและส่งค่ากลับมาเป็น JSON เมื่อค้นหา"""
        # ต้องจำลองการล็อกอินก่อน
        with self.client.session_transaction() as sess:
            sess['_user_id'] = '1'
            sess['_fresh'] = True

        response = self.client.get('/autocomplete?q=chicken')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.is_json)

    def test_search_displays_reviews(self):
        """Test if the search page successfully loads and displays the User Reviews section."""
        # Mock a logged-in session
        with self.client.session_transaction() as sess:
            sess['_user_id'] = '1'
            sess['_fresh'] = True

        # Perform a search request
        response = self.client.get('/search?q=chicken')
        self.assertEqual(response.status_code, 200)

        # Verify that the "User Reviews" header exists in the rendered HTML
        html_content = response.data.decode('utf-8')
        self.assertIn('User Reviews', html_content)

if __name__ == '__main__':
    unittest.main()
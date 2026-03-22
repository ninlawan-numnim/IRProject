from flask import Flask, render_template, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
# ตั้งค่าคีย์สำหรับการทำ Session และตั้งค่าเชื่อมต่อฐานข้อมูล SQLite
app = Flask(__name__)
app.config['SECRET_KEY'] = 'super_secret_key_se481'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///food_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' # ถ้ายังไม่ล็อกอินให้เด้งไปหน้า login
login_manager.login_message = "Please log in to access this page."

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # เช็กว่ามีชื่อผู้ใช้นี้ในระบบหรือยัง
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists. Please choose a different one.')
            return redirect(url_for('register'))

        # เข้ารหัสผ่านก่อนบันทึกลงฐานข้อมูล
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password_hash=hashed_password)

        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! You can now log in.')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        # ตรวจสอบชื่อผู้ใช้และรหัสผ่าน
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check your username and password.')

    return render_template('login.html')

@app.route('/logout')
@login_required # ต้องล็อกอินก่อนถึงจะล็อกเอาต์ได้
def logout():
    logout_user()
    return redirect(url_for('index'))

# สร้างตารางในฐานข้อมูลก่อนรันแอปพลิเคชัน
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    # รันเซิร์ฟเวอร์
    app.run(debug=True)
import re
import numpy as np
from spellchecker import SpellChecker
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import hashlib
import requests
from PIL import Image
from io import BytesIO
from flask import send_from_directory

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


class Folder(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    # เชื่อมความสัมพันธ์แบบ 1-to-Many ไปที่ Bookmark และตั้งค่าให้ลบบุ๊กมาร์กทิ้งด้วยถ้าโฟลเดอร์ถูกลบ
    bookmarks = db.relationship('Bookmark', backref='folder', lazy=True, cascade="all, delete-orphan")


class Bookmark(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    recipe_id = db.Column(db.Integer, nullable=False)
    recipe_name = db.Column(db.String(250), nullable=False)
    rating = db.Column(db.Integer, default=0)
    folder_id = db.Column(db.Integer, db.ForeignKey('folder.id'), nullable=False)

    # กฎ: 1 โฟลเดอร์ ห้ามมี recipe_id ซ้ำกัน
    __table_args__ = (
        db.UniqueConstraint('folder_id', 'recipe_id', name='unique_folder_recipe'),
    )

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

print("Loading dataset into RAM... This might take a few minutes.")
columns_to_load = [
    'RecipeId', 'Name', 'RecipeIngredientParts', 'RecipeInstructions',
    'Images', 'TotalTime', 'RecipeIngredientQuantities'
]
df = pd.read_parquet('data/recipes.parquet', columns=columns_to_load)

# จัดการค่าว่าง (NaN) และบังคับแปลงประเภทข้อมูลให้เป็น String (str) อย่างชัดเจน
df['Name'] = df['Name'].fillna('')

# นำ 3 คอลัมน์มาต่อกันเพื่อใช้ค้นหาพร้อมกัน
df['combined_text'] = df['Name'].astype(str) + " " + df['RecipeIngredientParts'].astype(str) + " " + df['RecipeInstructions'].astype(str)

# --- ฟังก์ชันจัดการข้อมูลเสริม ---
def format_time(pt_str):
    if not isinstance(pt_str, str):
        return 'N/A'
    if pt_str == 'NA':
        return 'N/A'

    h = re.search(r'(\d+)H', pt_str)
    m = re.search(r'(\d+)M', pt_str)
    h_str = f"{h.group(1)} Hr " if h else ""
    m_str = f"{m.group(1)} Min" if m else ""
    res = (h_str + m_str).strip()
    return res if res else 'N/A'


def extract_first_image(img_data):
    default_img = '/static/default_image.jpg'

    # กรณี 1: ข้อมูลเป็น List หรือ Array (จาก Parquet)
    if isinstance(img_data, (list, tuple)) or type(img_data).__name__ == 'ndarray':
        if len(img_data) > 0 and img_data[0] != 'character(0)':
            return str(img_data[0]).strip('"')
        return default_img

    # กรณี 2: ข้อมูลเป็น String ธรรมดา (จาก CSV)
    if isinstance(img_data, str):
        if img_data == 'character(0)' or img_data == 'NA' or img_data.strip() == '':
            return default_img
        match = re.search(r'(https?://[^\s",]+)', img_data)
        return match.group(1) if match else default_img

    return default_img


# สร้างคอลัมน์ใหม่ที่ผ่านการคลีนแล้ว
df['FormattedTime'] = df['TotalTime'].apply(format_time)
df['FirstImage'] = df['Images'].apply(extract_first_image)

print("Building TF-IDF Matrix...")
# สร้างโมเดล TF-IDF (ตัด stop words ภาษาอังกฤษออกเพื่อให้ผลลัพธ์แม่นยำขึ้น)
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
# เตรียมลิสต์ชื่ออาหารสำหรับระบบ Autocomplete
print("Building Vocabulary for Autocomplete...")
recipe_names_list = df['Name'].dropna().tolist()

print("System Ready!")


@app.route('/autocomplete')
@login_required
def autocomplete():
    q = request.args.get('q', '').lower()
    if len(q) < 2:
        return jsonify([])

    # ดึงชื่ออาหารที่มีคำค้นหาผสมอยู่ (ดึงมาแค่ 5 อันดับแรกเพื่อให้ทำงานเร็ว)
    suggestions = [name for name in recipe_names_list if q in str(name).lower()][:5]
    return jsonify(suggestions)

@app.route('/search', methods=['GET'])
@login_required
def search():
    query = request.args.get('q', '')
    results = []
    corrected_query = None

    if query:
        # --- ระบบตรวจคำผิด (Spell Correction) ---
        spell = SpellChecker()
        words = query.split()
        misspelled = spell.unknown(words)

        if misspelled:
            corrected_words = []
            for word in words:
                if word in misspelled:
                    correction = spell.correction(word)
                    corrected_words.append(correction if correction else word)
                else:
                    corrected_words.append(word)

            potential_correction = " ".join(corrected_words)
            # ถ้าคำที่แก้แล้วไม่เหมือนคำเดิม ให้ส่งไปโชว์ที่หน้าเว็บ
            if potential_correction.lower() != query.lower():
                corrected_query = potential_correction

        # --- ประมวลผล TF-IDF ด้วยคำค้นหาเดิม (query) ---
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # ดึง 20 อันดับแรก
        top_20_indices = similarities.argsort()[-20:][::-1]

        for idx in top_20_indices:
            if similarities[idx] > 0:
                recipe = df.iloc[idx]

                # ==========================================
                # [IR Feature 1: More Like This]
                # หาเมนูที่คล้ายกับเมนูนี้ โดยเทียบ Vector
                # ==========================================
                recipe_vec = tfidf_matrix[idx]
                sims = cosine_similarity(recipe_vec, tfidf_matrix).flatten()
                sim_indices = sims.argsort()[-4:][::-1]  # ดึง 4 อันดับ (อันดับ 1 คือตัวมันเอง)

                similar_recipes = []
                for s_idx in sim_indices:
                    if s_idx != idx and sims[s_idx] > 0:
                        similar_recipes.append(df.iloc[s_idx]['Name'])

                # ฟังก์ชันช่วยแปลง Array ให้เป็น List ของข้อความที่คลีนแล้ว
                def parse_array(arr):
                    if isinstance(arr, (list, tuple)) or type(arr).__name__ == 'ndarray':
                        return [str(x).strip() for x in arr if x is not None and str(x).strip() not in ('', 'None', 'nan')]
                    if pd.isna(arr):
                        return []
                    arr_str = str(arr)
                    matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", arr_str)
                    cleaned_items = [m[0] or m[1] for m in matches]
                    if not cleaned_items and arr_str.strip() not in ('', '[]', 'None'):
                        cleaned_items = [x.strip(' \'"[]') for x in arr_str.split(',')]
                    return [x for x in cleaned_items if x and x != 'None']

                # ดึงข้อมูลออกมาเป็น List
                parts = parse_array(recipe['RecipeIngredientParts'])
                qtys = parse_array(recipe['RecipeIngredientQuantities'])
                instructions = parse_array(recipe['RecipeInstructions'])

                # จับคู่ปริมาณ (Quantities) เข้ากับชื่อวัตถุดิบ (Parts)
                combined_ingredients = []
                for i in range(max(len(parts), len(qtys))):
                    q = qtys[i] if i < len(qtys) else ""
                    p = parts[i] if i < len(parts) else ""
                    if p.strip():
                        combined_ingredients.append(f"• {q} {p}".strip())

                # จัดฟอร์แมตขั้นตอนการทำ
                formatted_instructions = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(instructions)])

                results.append({
                    'id': recipe['RecipeId'],
                    'name': recipe['Name'],
                    'ingredients': "\n".join(combined_ingredients),
                    'instructions': formatted_instructions,
                    'image': recipe['FirstImage'],
                    'time': recipe['FormattedTime'],
                    'score': similarities[idx],
                    'similar_recipes': similar_recipes[:3] # <--- ส่งไปหน้าเว็บตรงนี้
                })

    folders = Folder.query.filter_by(user_id=current_user.id).all()

    return render_template('search.html', query=query, results=results, folders=folders,
                           corrected_query=corrected_query)

@app.route('/')
def index():
    if not current_user.is_authenticated:
        return render_template('index.html')

    summary_list = []
    category_list = []
    chosen_folder_name = None
    random_list = []

    # 1. Summary from all folders
    recent_bookmarks = Bookmark.query.join(Folder).filter(Folder.user_id == current_user.id).order_by(Bookmark.id.desc()).limit(4).all()
    for bm in recent_bookmarks:
        recipe_data = df[df['RecipeId'] == bm.recipe_id]
        if not recipe_data.empty:
            recipe = recipe_data.iloc[0]
            summary_list.append({'name': bm.recipe_name, 'image': recipe['FirstImage'], 'time': recipe['FormattedTime']})

    # 2. Selection from a specific chosen category
    user_folders = Folder.query.filter_by(user_id=current_user.id).all()
    if user_folders:
        chosen_folder = random.choice(user_folders)
        chosen_folder_name = chosen_folder.name
        folder_bms = Bookmark.query.filter_by(folder_id=chosen_folder.id).limit(4).all()
        for bm in folder_bms:
            recipe_data = df[df['RecipeId'] == bm.recipe_id]
            if not recipe_data.empty:
                recipe = recipe_data.iloc[0]
                category_list.append({'name': bm.recipe_name, 'image': recipe['FirstImage'], 'time': recipe['FormattedTime']})

    # 3. Completely random dishes
    random_recipes = df.sample(4)
    for _, recipe in random_recipes.iterrows():
        random_list.append({'name': recipe['Name'], 'image': recipe['FirstImage'], 'time': recipe['FormattedTime']})

    return render_template('index.html', summary_list=summary_list, category_list=category_list, chosen_folder_name=chosen_folder_name, random_list=random_list)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists. Please choose a different one.')
            return redirect(url_for('register'))

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

        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check your username and password.')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/create_folder', methods=['POST'])
@login_required
def create_folder():
    folder_name = request.form.get('folder_name')
    if folder_name:
        existing_folder = Folder.query.filter_by(name=folder_name, user_id=current_user.id).first()
        if not existing_folder:
            new_folder = Folder(name=folder_name, user_id=current_user.id)
            db.session.add(new_folder)
            db.session.commit()
            flash(f'folder has created "{folder_name}" successfully', 'success')
        else:
            flash('already has this folder name ', 'warning')
    return redirect(request.referrer or url_for('search'))

@app.route('/bookmark', methods=['POST'])
@login_required
def bookmark_recipe():
    recipe_id = request.form.get('recipe_id')
    recipe_name = request.form.get('recipe_name')
    folder_id = request.form.get('folder_id')

    if not folder_id:
        flash('please select foler before save', 'danger')
        return redirect(request.referrer or url_for('search'))

    existing_bookmark = Bookmark.query.filter_by(folder_id=folder_id, recipe_id=recipe_id).first()
    if existing_bookmark:
        flash(f'you save "{recipe_name}" in this folder already', 'warning')
    else:
        new_bookmark = Bookmark(recipe_id=recipe_id, recipe_name=recipe_name, folder_id=folder_id)
        db.session.add(new_bookmark)
        db.session.commit()
        flash(f'saved "{recipe_name}" already', 'success')

    return redirect(request.referrer or url_for('search'))

@app.route('/my_folders')
@login_required
def my_folders():
    folders = Folder.query.filter_by(user_id=current_user.id).all()
    return render_template('my_folders.html', folders=folders)


@app.route('/folder/<int:folder_id>')
@login_required
def folder_details(folder_id):
    folder = db.session.get(Folder, folder_id)
    if not folder or folder.user_id != current_user.id:
        flash('ไม่พบโฟลเดอร์ หรือคุณไม่มีสิทธิ์เข้าถึง', 'danger')
        return redirect(url_for('my_folders'))

    bookmarks = Bookmark.query.filter_by(folder_id=folder.id).all()
    recipe_list = []
    saved_recipe_ids = [bm.recipe_id for bm in bookmarks]

    for bm in bookmarks:
        recipe_data = df[df['RecipeId'] == bm.recipe_id]
        if not recipe_data.empty:
            recipe = recipe_data.iloc[0]
            recipe_list.append({
                'bookmark_id': bm.id,
                'recipe_id': bm.recipe_id,
                'name': bm.recipe_name,
                'rating': bm.rating,
                'image': recipe['FirstImage'],
                'time': recipe['FormattedTime']
            })

    # --- ระบบแนะนำอาหาร (Recommendation System) ---
    recommendations = []
    if saved_recipe_ids:
        saved_indices = df.index[df['RecipeId'].isin(saved_recipe_ids)].tolist()

        if saved_indices:
            saved_vectors = tfidf_matrix[saved_indices]
            profile_vector = np.asarray(saved_vectors.mean(axis=0))
            similarities = cosine_similarity(profile_vector, tfidf_matrix).flatten()
            top_indices = similarities.argsort()[::-1]

            for idx in top_indices:
                rec_id = df.iloc[idx]['RecipeId']
                if rec_id not in saved_recipe_ids:
                    recipe = df.iloc[idx]
                    recommendations.append({
                        'id': recipe['RecipeId'],
                        'name': recipe['Name'],
                        'image': recipe['FirstImage'],
                        'time': recipe['FormattedTime'],
                        'score': similarities[idx]
                    })
                if len(recommendations) >= 4:
                    break

    return render_template('folder_details.html', folder=folder, recipes=recipe_list, recommendations=recommendations)

@app.route('/update_rating/<int:bookmark_id>', methods=['POST'])
@login_required
def update_rating(bookmark_id):
    bookmark = db.session.get(Bookmark, bookmark_id)
    if bookmark and bookmark.folder.user_id == current_user.id:
        new_rating = request.form.get('rating', type=int)
        if new_rating in [1, 2, 3, 4, 5]:
            bookmark.rating = new_rating
            db.session.commit()
            flash('updated rating', 'success')
    return redirect(request.referrer or url_for('my_folders'))

@app.route('/delete_bookmark/<int:bookmark_id>', methods=['POST'])
@login_required
def delete_bookmark(bookmark_id):
    bookmark = db.session.get(Bookmark, bookmark_id)
    if bookmark and bookmark.folder.user_id == current_user.id:
        db.session.delete(bookmark)
        db.session.commit()
        flash('removed from folder', 'success')
    return redirect(request.referrer or url_for('my_folders'))

@app.route('/delete_folder/<int:folder_id>', methods=['POST'])
@login_required
def delete_folder(folder_id):
    folder = db.session.get(Folder, folder_id)
    if folder and folder.user_id == current_user.id:
        db.session.delete(folder)
        db.session.commit()
        flash(f'folder deleted {folder.name} successfully', 'success')
    return redirect(url_for('my_folders'))


@app.route('/all_bookmarks')
@login_required
def all_bookmarks():
    all_bms = Bookmark.query.join(Folder).filter(Folder.user_id == current_user.id).order_by(
        Bookmark.rating.desc()).all()

    bookmark_list = []
    for bm in all_bms:
        recipe_data = df[df['RecipeId'] == bm.recipe_id]
        if not recipe_data.empty:
            recipe = recipe_data.iloc[0]
            bookmark_list.append({
                'name': bm.recipe_name,
                'rating': bm.rating,
                'folder_name': bm.folder.name,
                'image': recipe['FirstImage'],
                'time': recipe['FormattedTime']
            })
    return render_template('all_bookmarks.html', bookmarks=bookmark_list)


CACHE_DIR = os.path.join(app.root_path, 'static', 'image_cache')
os.makedirs(CACHE_DIR, exist_ok=True)


@app.route('/cached_image')
def cached_image():
    image_url = request.args.get('url')

    if not image_url or image_url.startswith('/static/'):
        return redirect(image_url or '/static/default_image.jpg')

    url_hash = hashlib.md5(image_url.encode('utf-8')).hexdigest()
    filename = f"{url_hash}.jpg"
    filepath = os.path.join(CACHE_DIR, filename)

    if os.path.exists(filepath):
        return send_from_directory(CACHE_DIR, filename, max_age=31536000)

    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))

        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        base_width = 400
        if img.width > base_width:
            wpercent = (base_width / float(img.width))
            hsize = int((float(img.height) * float(wpercent)))
            img = img.resize((base_width, hsize), Image.Resampling.LANCZOS)

        img.save(filepath, "JPEG", quality=85)
        return send_from_directory(CACHE_DIR, filename, max_age=31536000)

    except Exception as e:
        print(f"Error caching image: {e}")
        return redirect('/static/default_image.jpg')

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
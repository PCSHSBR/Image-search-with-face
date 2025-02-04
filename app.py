import streamlit as st
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
import os
from pathlib import Path
import torch
import random
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get admin credentials from environment variables
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# Initialize face analyzer with caching
@st.cache_resource
def init_face_analyzer():
    # ตรวจสอบการใช้งาน MPS
    if torch.backends.mps.is_available():
        st.sidebar.success("Using M2 GPU (Metal)")
        # ใช้ MPS backend
        analyzer = FaceAnalysis(
            name="buffalo_l",
            providers=['MPSExecutionProvider'],
            allowed_modules=['detection', 'recognition']  # เพิ่มโมดูลที่ใช้ GPU
        )
        analyzer.prepare(ctx_id=0)  # ใช้ GPU
    else:
        st.sidebar.info("Using CPU")
        # Fallback to CPU
        analyzer = FaceAnalysis(
            name="buffalo_l",
            providers=['CPUExecutionProvider']
        )
        analyzer.prepare(ctx_id=-1)
    
    return analyzer

# Cache database loading
@st.cache_data
def load_database_images(database_dir, _face_analyzer):
    database = []
    
    # รวบรวมรูปทั้งหมด
    all_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        all_images.extend(list(Path(database_dir).glob(ext)))
    
    # สุ่มเลือกรูป 70% ของรูปทั้งหมด
    sample_size = int(len(all_images) * 0.7)
    if sample_size > 0:
        selected_images = random.sample(all_images, sample_size)
    else:
        selected_images = all_images
    
    # Process images
    for img_path in selected_images:
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                # Resize ให้เล็กลงเพื่อเร็วขึ้น
                img = preprocess_image(img, max_size=320)  # ลดขนาดลง
                faces = _face_analyzer.get(img)
                if faces:
                    database.append({
                        "path": str(img_path),
                        "embeddings": [face.normed_embedding for face in faces]
                    })
        except Exception as e:
            st.sidebar.error(f"Error loading {img_path}: {str(e)}")
            continue
    
    return database

# Optimize image size
def preprocess_image(image, max_size=640):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size)
    return image

# Vectorized matching
@st.cache_data
def find_matches_vectorized(query_embedding, database, threshold):
    if not database or query_embedding is None:
        return []
    
    # Combine all embeddings into a single matrix
    all_embeddings = []
    all_paths = []
    for db in database:
        if db.get("embeddings"):  # ตรวจสอบว่ามี embeddings
            all_embeddings.extend(db["embeddings"])
            all_paths.extend([db["path"]] * len(db["embeddings"]))
    
    if not all_embeddings:
        return []
        
    try:
        # Calculate similarities in one go
        all_embeddings = np.vstack(all_embeddings)
        similarities = np.dot(all_embeddings, query_embedding)
        
        # Find matches above threshold
        mask = similarities > threshold
        matches = [{
            "path": path,
            "similarity": float(sim)
        } for path, sim in zip(np.array(all_paths)[mask], similarities[mask])]
        
        return sorted(matches, key=lambda x: x["similarity"], reverse=True)
    except Exception as e:
        st.error(f"Error in matching: {str(e)}")
        return []

# Efficient result display
def display_results(matches, cols=3):
    if not matches:
        return
    
    # กรองเฉพาะ matches ที่มีความคล้ายเกิน 25%
    filtered_matches = [m for m in matches if m.get('similarity', 0) * 100 >= 25]
    
    if not filtered_matches:
        st.write("No matches above 25% similarity")
        return
    
    # แสดงจำนวนรูปที่เจอ
    st.write(f"Found {len(filtered_matches)} matches above 25% similarity")
    
    # แบ่งการแสดงผลเป็นแถว แถวละ 3 รูป
    for i in range(0, len(filtered_matches), cols):
        columns = st.columns(cols)
        for j, match in enumerate(filtered_matches[i:i+cols]):
            with columns[j]:
                similarity_percent = f"{match['similarity']*100:.1f}%"
                match_img = cv2.imread(match["path"])
                if match_img is not None:
                    match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
                    st.image(match_img, caption=f"Similarity: {similarity_percent}")

def create_new_album(album_name):
    album_dir = os.path.join("albums", album_name)
    if not os.path.exists(album_dir):
        os.makedirs(album_dir)
        st.success(f"Created new album: {album_name}")
    else:
        st.warning(f"Album '{album_name}' already exists")

def upload_to_album(album_name, uploaded_files, face_analyzer):
    album_dir = os.path.join("albums", album_name)
    if not os.path.exists(album_dir):
        st.error(f"Album '{album_name}' does not exist")
        return
    
    uploaded_count = 0
    for uploaded_file in uploaded_files:
        file_path = os.path.join(album_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_count += 1
    
    st.success(f"Uploaded {uploaded_count} files to {album_name}")
    return reload_database(album_name, face_analyzer)

def upload_folder_to_album(album_name, folder_path, face_analyzer):
    album_dir = os.path.join("albums", album_name)
    if not os.path.exists(album_dir):
        st.error(f"Album '{album_name}' does not exist")
        return
    
    if not os.path.isdir(folder_path):
        st.error(f"Invalid folder path: {folder_path}")
        return
    
    uploaded_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(album_dir, file)
                shutil.copy(src_path, dest_path)
                uploaded_files.append(file)
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} files to {album_name}")
    else:
        st.warning("No valid image files found in the folder")
    return reload_database(album_name, face_analyzer)

def view_album_images(album_name):
    album_dir = os.path.join("albums", album_name)
    if not os.path.exists(album_dir):
        st.error(f"Album '{album_name}' does not exist")
        return
    
    images = [f for f in os.listdir(album_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        st.warning(f"No images found in album '{album_name}'")
        return
    
    st.write(f"Found {len(images)} images in album '{album_name}'")
    
    # แสดงรูปภาพในแถวละ 3 รูป
    cols = 3
    for i in range(0, len(images), cols):
        columns = st.columns(cols)
        for j, img_name in enumerate(images[i:i+cols]):
            with columns[j]:
                img_path = os.path.join(album_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(img, caption=img_name)

def check_admin(username, password):
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

def reload_database(album_name, face_analyzer):
    database_dir = os.path.join("albums", album_name)
    return load_database_images(database_dir, face_analyzer)

def main():
    st.set_page_config(
        page_title="Face Recognition",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("Face Recognition App 👤")
    
    # Initialize components
    status = st.empty()
    progress = st.progress(0)
    face_analyzer = init_face_analyzer()
    
    # Login section
    st.sidebar.header("Admin Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    is_admin = check_admin(username, password)
    
    if is_admin:
        st.sidebar.success("Logged in as Admin")
        
        # Create new album section
        st.sidebar.header("Create New Album")
        new_album_name = st.sidebar.text_input("Enter album name")
        if st.sidebar.button("Create Album"):
            if new_album_name:
                create_new_album(new_album_name)
            else:
                st.sidebar.warning("Please enter an album name")
        
        # Upload to album section
        st.sidebar.header("Upload to Album")
        album_names = [name for name in os.listdir("albums") if os.path.isdir(os.path.join("albums", name))]
        selected_album = st.sidebar.selectbox("Select album", album_names, key="select_album_upload")
        # Multiple file upload
        uploaded_files = st.sidebar.file_uploader(
            "Choose images", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )
        if uploaded_files and selected_album:
            database = upload_to_album(selected_album, uploaded_files, face_analyzer)
        
        # Folder upload
        folder_path = st.sidebar.text_input("Enter folder path to upload")
        if st.sidebar.button("Upload Folder") and selected_album and folder_path:
            database = upload_folder_to_album(selected_album, folder_path, face_analyzer)
    else:
        st.sidebar.warning("Please login as Admin to create albums and upload images")
    
    # Load database from selected album
    status.text("Loading database...")
    album_names = [name for name in os.listdir("albums") if os.path.isdir(os.path.join("albums", name))]
    selected_album = st.sidebar.selectbox("Select album", album_names, key="select_album_main")
    database_dir = os.path.join("albums", selected_album)  # ใช้โฟลเดอร์ของ album ที่เลือก
    Path(database_dir).mkdir(exist_ok=True)
    database = load_database_images(database_dir, face_analyzer)
    
    # UI Controls
    with st.sidebar:
        st.write(f"Database images: {len(database)}")
        similarity_threshold = st.slider(
            "Similarity Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.25
        )
        
        # เพิ่มปุ่มดูรูปใน albums
        if st.sidebar.button("View Album Images"):
            view_album_images(selected_album)

    status.text("Ready for image upload")
    progress.progress(100)
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            status.text("Processing image...")
            progress.progress(30)
            
            # Read and preprocess image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = preprocess_image(image)
            
            # Detect faces
            status.text("Detecting faces...")
            faces = face_analyzer.get(image)
            progress.progress(60)
            
            if len(faces) > 0:
                status.text(f"Found {len(faces)} faces")
                
                # Create columns for results
                result_cols = st.columns(min(len(faces), 3))
                
                for i, (face, col) in enumerate(zip(faces, result_cols)):
                    with col:
                        # Show face info
                        gender = "Male" if face.sex == 1 else "Female"
                        age = int(face.age) if face.age is not None else 0
                        st.write(f"Face #{i+1}: {gender}, {age} years")
                        
                        # Find and display matches
                        matches = find_matches_vectorized(
                            face.normed_embedding, 
                            database, 
                            similarity_threshold
                        )
                        display_results(matches)
                
                status.text("Processing complete!")
                progress.progress(100)
                
            else:
                status.text("No faces detected")
                progress.progress(100)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            progress.progress(100)

if __name__ == "__main__":
    main() 
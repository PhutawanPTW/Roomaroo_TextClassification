from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import numpy as np
from scipy.sparse import hstack, csr_matrix
from scipy import sparse
import logging
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_negations, thai_stopwords
import re
import string
import emoji
from collections import Counter
from utils import custom_tokenizer

# สร้าง FastAPI app (เปิด debug เพื่อช่วยตรวจปัญหาในช่วงพัฒนา)
app = FastAPI(debug=True)

# เพิ่ม CORS เพื่อให้ Angular frontend สามารถเชื่อมต่อได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "https://your-website-domain.com"],  # URL ของ Angular app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# กำหนดโครงสร้างข้อมูลสำหรับรับรีวิวจาก frontend
class ReviewInput(BaseModel):
    text: str

# ฟังก์ชันทำความสะอาดข้อความ
def clean_text(text):
    if not isinstance(text, str):
        return ""
    thai_stop_words = list(thai_stopwords())
    important_words = [
        # คำปฏิเสธและเน้นความหมาย
        "ไม่", "ไม่มี", "ไม่ได้", "ไม่ค่อย", "ไม่เคย", "ห้าม", "ยังไม่", "ไม่ยอม",
        
        # คำแสดงความรู้สึก/คุณภาพ (เชิงบวก)
        "ดี", "ดีมาก", "สะอาด", "เย็น", "กว้าง", "ใหม่", "สวย", "น่าอยู่", "สบาย", "ชอบ", 
        "ประทับใจ", "สะดวก", "ปลอดภัย", "คุ้ม", "คุ้มค่า", "เงียบ", "ครบ", "พอใจ", "เร็ว",
        "โอเค", "โอเคเลย", "เยี่ยม", "ถูกใจ", "ทำเลดี", "ใกล้", "ครบครัน",
        
        # คำแสดงความรู้สึก/คุณภาพ (เชิงลบ)
        "แย่", "ไม่ดี", "เหม็น", "ร้อน", "แคบ", "เก่า", "สกปรก", "พัง", "เสียงดัง", "แพง",
        "ไม่ชอบ", "แออัด", "รก", "อันตราย", "ช้า", "ผิดหวัง", "ห่วย", "เฟล", "กาก",
        "ไม่คุ้ม", "ไกล", "รั่ว", "ทรุด", "ทรุดโทรม", "เสื่อม",
        
        # คำแสดงระดับความเข้ม
        "มาก", "สุดๆ", "เยอะ", "น้อย", "ที่สุด", "หลาย", "ทุก", "เกิน", "จัด", "โคตร",
        "มากๆ", "สุดยอด", "ธรรมดา", "พอใช้", "บ่อย", "ตลอด", "เวอร์",
        
        # อุปกรณ์/สิ่งอำนวยความสะดวก
        "แอร์", "น้ำ", "ไฟ", "ห้องน้ำ", "เตียง", "ฝักบัว", "เน็ต", "ไวไฟ", "ไฟฟ้า", "ประปา",
        "เฟอร์", "ลิฟต์", "ที่จอด", "จอดรถ", "ซักผ้า", "ตู้เย็น", "ทีวี", "จาน", "ไมโครเวฟ", 
        "เตา", "น้ำอุ่น", "ผ้าปู", "โต๊ะ", "เก้าอี้", "ตู้", "ชั้นวาง", "ปลั๊ก", "สัญญาณ",
        
        # สิ่งแวดล้อม
        "เสียง", "มด", "แมลง", "แมลงสาบ", "หนู", "ยุง", "ฝุ่น", "กลิ่น", "เพื่อนบ้าน",
        "ข้างห้อง", "ข้างนอก", "ถนน", "ทางเดิน", "ลานจอด", "ชั้นบน", "บันได", "กำแพง",
        
        # บริการ/การจัดการ
        "ดูแล", "บริการ", "ซ่อม", "แก้ไข", "จัดการ", "พนักงาน", "แม่บ้าน", "รปภ", "เจ้าของ",
        "นิติ", "กฎ", "ระเบียบ", "ค่าเช่า", "ค่าไฟ", "ค่าน้ำ", "ค่าส่วนกลาง", "มัดจำ", "ประกัน", 
        "สัญญา", "ฝากของ", "รับพัสดุ", "คีย์การ์ด", "ล็อค", "รอนาน", "ไม่มาดู", "ไม่ซ่อม",
        
        # คำเชื่อมที่สำคัญในการแสดงความคิดเห็น
        "แต่", "แต่ว่า", "ถึงแม้", "อย่างไรก็ตาม", "เพราะ", "เพราะว่า", "เนื่องจาก", "คือ", "ก็คือ",
        "ส่วน", "นอกจากนี้", "ที่จริง", "จริงๆ", "ก็", "แม้", "ที่", "ตอนแรก", "พอดี", "แล้วก็"
    ]
    custom_stop_words = [word for word in thai_stop_words if word not in important_words]
    punct = string.punctuation.replace('!', '').replace('?', '').replace('.', '')
    text = emoji.replace_emoji(text, replace="")
    text = ''.join(char if char not in punct else ' ' for char in text)
    text = re.sub(r'([ก-๙a-zA-Z])\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', " ", text).strip().lower()
    words = word_tokenize(text, engine='newmm')
    words = [word for word in words if word not in custom_stop_words]
    text = ' '.join(words)
    return text

# ฟังก์ชันสกัดคุณลักษณะเพิ่มเติม
def extract_features(text):
    words = word_tokenize(text, engine='newmm')
    word_count = len(words)
    features = {
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'sentence_count': text.count('.') + 1,
        'word_count': word_count,
        'avg_word_length': sum(len(word) for word in words) / max(word_count, 1),
        'text_length': len(text)
    }

    # ===== เพิ่มส่วนนี้ =====
    # นับจำนวนครั้งที่มีการพิมพ์ตัวอักษรซ้ำ 3 ตัวขึ้นไป (เช่น "มากกกก")
    repeated_chars_pattern = re.findall(r'([ก-๙a-zA-Z])\1{2,}', text)
    features['repeated_chars_count'] = len(repeated_chars_pattern)
    
    # นับจำนวนตัวอักษรซ้ำทั้งหมด (รวมความยาวของการซ้ำ)
    total_repeated_length = sum(len(match) + 1 for match in repeated_chars_pattern)
    features['repeated_chars_intensity'] = total_repeated_length
    # ===== จบส่วนที่เพิ่ม =====

    word_counts = Counter(words)
    repeated_words = sum(1 for count in word_counts.values() if count > 1)
    features['repeated_words_ratio'] = repeated_words / max(word_count, 1)
    negation_words = thai_negations()
    features['negation_count'] = sum(1 for word in words if word in negation_words)
    features['punctuation_ratio'] = len([c for c in text if c in string.punctuation]) / max(len(text), 1)
    features['words_per_sentence'] = word_count / max(features['sentence_count'], 1)
    return features

# ฟังก์ชันสร้างเวกเตอร์ประโยคจาก Thai2fit
def enhanced_sentence_vectorizer(text, dim=300):
    global thai2fit
    words = word_tokenize(text, engine="newmm")
    vec = np.zeros(dim)
    word_count = 0
    total_words = len(words)
    for i, word in enumerate(words):
        if word.strip() and word != " ":
            position_weight = 1.0 + (i / max(total_words, 1)) * 0.5
            if word in thai2fit:
                vec += thai2fit[word] * position_weight
                word_count += 1
            else:
                char_vec, char_count = np.zeros(dim), 0
                for char in word:
                    if char in thai2fit:
                        char_vec += thai2fit[char]
                        char_count += 1
                if char_count > 0:
                    vec += (char_vec / char_count) * position_weight
                    word_count += 1
    return vec / max(word_count, 1)

# โหลดโมเดลและส่วนประกอบเมื่อเริ่มต้น API
@app.on_event("startup")
def load_model_and_components():
    global model, tfidf_vec, count_vec, scaler_vec, features_scaler_vec, thai2fit
    output_dir = "saved_models"
    try:
        model = joblib.load(os.path.join(output_dir, "best_model.pkl"))
        tfidf_vec = joblib.load(os.path.join(output_dir, "tfidf_vectorizer.pkl"))
        count_vec = joblib.load(os.path.join(output_dir, "count_vectorizer.pkl"))
        scaler_vec = joblib.load(os.path.join(output_dir, "scaler.pkl"))
        features_scaler_vec = joblib.load(os.path.join(output_dir, "features_scaler.pkl"))
        thai2fit = joblib.load(os.path.join(output_dir, "thai2fit_model.pkl"))
        print("โหลดโมเดลและส่วนประกอบทั้งหมดเรียบร้อย")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ไม่สามารถโหลดโมเดลได้: {str(e)}")

# API endpoint สำหรับทำนายคะแนนรีวิว
@app.post("/predict")
async def predict_review(review: ReviewInput):
    try:
        text = review.text if isinstance(review.text, str) else ""
        if not isinstance(text, str):
            raise HTTPException(status_code=400, detail="รูปแบบข้อมูลไม่ถูกต้อง ต้องเป็นสตริงใน key 'text'")
        text = text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="กรุณากรอกข้อความรีวิว")

        # ทำความสะอาดข้อความ
        cleaned_text = clean_text(text)

        # สกัดคุณลักษณะเพิ่มเติม
        features = extract_features(cleaned_text)

        # สร้าง features สำหรับการทำนาย
        logger = logging.getLogger("uvicorn.error")

        if model.__class__.__name__ == 'MultinomialNB':
            review_tfidf = tfidf_vec.transform([cleaned_text])
            review_count = count_vec.transform([cleaned_text])
            logger.info(f"NB pipeline shapes tfidf={review_tfidf.shape}, count={review_count.shape}")
            review_combined = hstack([review_tfidf, review_count])
        else:
            review_vector = enhanced_sentence_vectorizer(cleaned_text).reshape(1, -1)
            review_vector_sparse = csr_matrix(review_vector)
            review_vector_scaled = scaler_vec.transform(review_vector_sparse)
            # ถ้า scaler คืน dense ndarray ให้แปลงกลับเป็น csr เพื่อใช้ร่วมกับ hstack
            if not sparse.issparse(review_vector_scaled):
                review_vector_scaled = csr_matrix(review_vector_scaled)
            review_tfidf = tfidf_vec.transform([cleaned_text])
            review_count = count_vec.transform([cleaned_text])
            additional_features = np.array([[
                features['exclamation_count'], features['question_count'], features['sentence_count'],
                features['word_count'], features['avg_word_length'], features['repeated_words_ratio'],
                features['negation_count'], features['punctuation_ratio'], features['text_length'],
                features['words_per_sentence'], features['repeated_chars_count'], features['repeated_chars_intensity']
            ]])
            additional_features_scaled = features_scaler_vec.transform(additional_features)
            if not sparse.issparse(additional_features_scaled):
                additional_features_scaled = csr_matrix(additional_features_scaled)

            logger.info(
                f"shapes tfidf={review_tfidf.shape}, count={review_count.shape}, vec_scaled={getattr(review_vector_scaled, 'shape', None)}, add_scaled={getattr(additional_features_scaled, 'shape', None)}"
            )
            review_combined = hstack([review_tfidf, review_count, review_vector_scaled, additional_features_scaled])

        # ทำนายคะแนน
        predicted_rating = int(model.predict(review_combined)[0])
        probabilities = model.predict_proba(review_combined)[0].tolist() if hasattr(model, 'predict_proba') else None

        # เตรียม response
        response = {
            "review": text,
            "cleaned_review": cleaned_text,
            "predicted_rating": predicted_rating,
            "confidence": float(max(probabilities)) if probabilities else None,
            "probabilities": {f"{i}": float(prob) for i, prob in enumerate(probabilities, 1)} if probabilities else None
        }
        return response

    except Exception as e:
        logging.getLogger("uvicorn.error").exception("predict error")
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")
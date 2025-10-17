import sys
import re
import json
import pandas as pd

# Fixed target CSV
TARGET_CSV = 'Data/dorm_reviews.csv'

# Heuristic column candidates
TEXT_CANDIDATES = ['review', 'text', 'comment', 'รีวิว', 'ข้อความ']
RATING_CANDIDATES = ['rating', 'score', 'stars', 'คะแนน', 'ดาว']

# Desired profanity share by class (tunable)
TARGET_RATE = {1: 0.70, 2: 0.50, 3: 0.03}

PROF_WORDS = [
	'ควย','เหี้ย','สัส','ไอสัส','เชี่ย','เชี้ย','ควาย','ชิบหาย',
	'ส้นตีน','แม่ง','เวร','บัดซบ','ห่า','อีดอก','ระยำ','เฮงซวย','อัปรีย์','สถุน'
]

def build_pattern(words):
	escaped = [re.escape(w) for w in words]
	pattern = '|'.join(escaped)
	return re.compile(pattern, flags=re.IGNORECASE)

PROF_RE = build_pattern(PROF_WORDS)

def pick_col(cols, candidates):
	for c in cols:
		n = c.lower()
		if any(k in n for k in candidates):
			return c
	return None

def read_csv_any(path):
	for enc in ('utf-8', 'utf-8-sig', 'cp874', 'latin1'):
		try:
			return pd.read_csv(path, encoding=enc)
		except Exception:
			continue
	raise RuntimeError(f'อ่านไฟล์ไม่ได้: {path}')

def normalize_text(s: str) -> str:
	s = s.lower()
	s = re.sub(r'\s+', '', s)
	s = re.sub(r'(.)\1{2,}', r'\1\1', s)
	return s

def is_profane(s: str) -> bool:
	if not isinstance(s, str):
		return False
	s = normalize_text(s)
	return bool(PROF_RE.search(s))

def find_profane_words(s: str):
	"""คืนรายการคำหยาบ (ฐานคำจาก PROF_WORDS) ที่พบในข้อความสตริง s"""
	if not isinstance(s, str):
		return []
	n = normalize_text(s)
	found = []
	for w in PROF_WORDS:
		if re.search(re.escape(w), n, flags=re.IGNORECASE):
			found.append(w)
	# คืนแบบไม่ซ้ำและคงลำดับ
	seen = set()
	unique = []
	for w in found:
		if w not in seen:
			unique.append(w)
			seen.add(w)
	return unique

def main():
	sys.stdout.reconfigure(encoding='utf-8')
	df = read_csv_any(TARGET_CSV)
	text_col = pick_col(df.columns, TEXT_CANDIDATES)
	rating_col = pick_col(df.columns, RATING_CANDIDATES)
	if not text_col or not rating_col:
		print(f'หา column ข้อความ/คะแนนไม่เจอ -> {list(df.columns)}')
		return

	sub = df[[text_col, rating_col]].dropna()
	sub[rating_col] = pd.to_numeric(sub[rating_col], errors='coerce')
	sub = sub.dropna()

	sub['profane'] = sub[text_col].apply(is_profane)
	grp = sub.groupby(rating_col)['profane'].agg(total='count', profane_count='sum')
	grp['profane_rate'] = grp['profane_count'] / grp['total']

	rows = []
	for k, row in grp.iterrows():
		k_int = int(k)
		target = TARGET_RATE.get(k_int)
		need = None
		if target is not None:
			total = int(row['total'])
			prof = int(row['profane_count'])
			target_prof = int(round(target * total))
			need = max(0, target_prof - prof)
		rows.append({
			'rating': k_int,
			'total': int(row['total']),
			'profane_count': int(row['profane_count']),
			'profane_rate': round(float(row['profane_rate']), 4),
			'add_profane_needed': need
		})

	print('ไฟล์:', TARGET_CSV)
	print(pd.DataFrame(rows).sort_values('rating').to_string(index=False))

	# แสดงรายการรีวิวที่ถูกจัดว่าเป็นคำหยาบ (พิมพ์ทั้งหมด ไม่บันทึกไฟล์)
	prof_df = sub[sub['profane']].copy()
	prof_df['profane_words'] = prof_df[text_col].apply(find_profane_words)
	if not prof_df.empty:
		prof_df = prof_df.reset_index().rename(columns={'index':'row_index'})
		view_cols = ['row_index', rating_col, 'profane_words', text_col]
		print('\nรายการรีวิวที่เป็นคำหยาบทั้งหมด (แสดงคำหยาบที่พบในแต่ละรีวิว):')
		print(prof_df[view_cols].to_string(index=False))
	else:
		print('\nไม่พบบรรทัดที่เป็นคำหยาบตามลิสต์ที่ตั้งไว้')

	print('\nJSON:', json.dumps(rows, ensure_ascii=False, indent=2))

if __name__ == '__main__':
	import pandas as pd  # ensure imported for type usage above
	main()



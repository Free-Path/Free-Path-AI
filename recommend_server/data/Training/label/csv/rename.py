#%%
import os
import re

# 폴더 경로 설정
folder_path = './'  # 여기에 실제 폴더 경로를 입력하세요

# 파일 이름에서 깨진 문자 제거 (점은 제외)하는 함수
def clean_filename(filename):
    # 파일 이름에서 한글, 영어, 숫자, 밑줄, 하이픈, 점만 남기기
    clean_name = re.sub(r'[^a-zA-Z0-9가-힣_. -]', '', filename)
    return clean_name

# 폴더 내 모든 파일 확인
for filename in os.listdir(folder_path):
    old_file_path = os.path.join(folder_path, filename)

    # 새로운 파일 이름 생성 (깨진 문자 제거)
    new_filename = clean_filename(filename)

    # 새 파일 경로 생성
    new_file_path = os.path.join(folder_path, new_filename)

    # 기존 파일명과 새로운 파일명이 다를 경우에만 이름 변경
    if old_file_path != new_file_path:
        try:
            # 파일 이름 변경
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: '{filename}' -> '{new_filename}'")
        except Exception as e:
            print(f"Error renaming file '{filename}': {e}")
    else:
        print(f"No change needed for '{filename}'")

print("파일 이름에서 깨진 문자가 제거되었습니다.")


# 폴더 내 모든 파일 확인
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        # 파일 읽기 (UTF-8 인코딩)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 파일 다시 쓰기 (UTF-8 with BOM 인코딩)
        with open(file_path, 'w', encoding='utf-8-sig') as file:
            file.write(content)

print("모든 CSV 파일이 UTF-8(BOM) 인코딩으로 변환되었습니다.")

# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
# --------------------------------------------------------------
# 1) 기본 라이브러리 불러오기
# --------------------------------------------------------------
import os
import glob

# ==============================================================
# 0. 변수 정의
# ==============================================================
base_path = './Dataset/Golf_Ball_Split/labels'

# ==============================================================
# 1. 폴더 내 파일명 중복 확인
# ==============================================================
# --------------------------------------------------------------
# 1) Label 파일명 추출 + Label 수정
# --------------------------------------------------------------
def revise_label(labels_path):
    # (1) Label 파일명 추출
    label = None
    for label_path in glob.glob(os.path.join(labels_path, '*.txt')):
        with open(label_path, 'r') as f:
            # (2) Label 한줄씩 불러오기
            for line in f.readlines():
                # 1] Label Split
                # print(line.split(' ', maxsplit=1))
                label, bbox = line.split(' ', maxsplit=1)
                # print(f'label : {label}')

                # 2] Label 자료형 변환
        with open(label_path, 'w') as f:
            # (3) Label 변환
            # print(f'label + bbox : 1 {bbox}')
            f.write('0 ' + bbox)

for idx, f_path in enumerate(['train/', 'val/', 'test/']):
    print(f_path)
    revise_label(f'{base_path}/{f_path}')
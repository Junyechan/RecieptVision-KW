"""Run: python test_top_safety_v5_1.py
Put next to preprocessing_v5_1.py and the images/ directory.
"""
from pathlib import Path
import csv
from preprocessing import preprocess_receipt, save_image

BASE=Path(__file__).resolve().parent
NAMES=['receipt_01.jpg','receipt_02.jpg','receipt_03.jpg','receipt_04.jpg','receipt_05.jpg','sample_receipt.jpg']
OUT=BASE/'outputs'/'pipeline_v5_1'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in NAMES:
        path=BASE/'images'/name
        if not path.exists():
            print('[MISSING]',name)
            continue
        r=preprocess_receipt(path)
        print('[OK]' if r['success'] else '[FAIL]',name,r['top_correction'],r['error'])
        if not r['success']:
            continue
        folder=OUT/path.stem
        folder.mkdir(parents=True,exist_ok=True)
        save_image(folder/'warped.png',r['warped'])
        save_image(folder/'ocr_ready.png',r['ocr_image'])
        save_image(folder/'detection_overlay.png',r['detection_overlay'])
        changed=r['ocr_corners'] != r['corners']
        rows.append({'filename':name,'top_correction':r['top_correction'],
                     'ocr_crop_expanded':changed,'detection_score':r['detection_score'],
                     'corners':str(r['corners']),'ocr_corners':str(r['ocr_corners'])})
    if rows:
        with (OUT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    print('성공:',len(rows),'/',len(NAMES),'결과:',OUT)

if __name__=='__main__':main()

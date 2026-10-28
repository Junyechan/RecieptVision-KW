from pathlib import Path
import csv
import json
import cv2
import numpy as np
from preprocessing_v5 import preprocess_receipt, read_image, save_image

BASE=Path(__file__).resolve().parent
IMAGES=BASE/'images'
OUT=BASE/'outputs'/'pipeline_v5'
NAMES=['receipt_01.jpg','receipt_02.jpg','receipt_03.jpg','receipt_04.jpg','receipt_05.jpg','sample_receipt.jpg']

def polygon_iou(a,b):
    a=np.asarray(a,np.float32).reshape(4,2)
    b=np.asarray(b,np.float32).reshape(4,2)
    overlap,_=cv2.intersectConvexConvex(a,b)
    union=abs(cv2.contourArea(a))+abs(cv2.contourArea(b))-overlap
    return float(overlap/union) if union>0 else 0.0

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for filename in NAMES:
        path=IMAGES/filename
        if not path.exists():
            print('[MISSING]',path)
            rows.append({'image':filename,'success':False,'error':'missing image'})
            continue
        result=preprocess_receipt(path)
        row={'image':filename,'success':result['success'],'error':result['error'] or '',
             'source':result['detection_source'],'score':result['detection_score'],
             'candidate_count':result['candidate_count'],'iou_gt':''}
        folder=OUT/path.stem;folder.mkdir(parents=True,exist_ok=True)
        if result['success']:
            for key,name in [('detection_overlay','detection_overlay.png'),('warped','warped.png'),('ocr_image','ocr_ready.png')]:
                save_image(folder/name,result[key])
            for key,image in result['variants'].items():save_image(folder/(key+'.png'),image)
            top=result['top_candidates']
            original=read_image(path)
            for c in top:
                overlay=original.copy()
                pts=np.rint(c['corners']).astype(np.int32)
                cv2.polylines(overlay,[pts],True,(0,255,0),3)
                save_image(folder/f"candidate_{c['rank']:02d}.png",overlay)
            (folder/'result.json').write_text(json.dumps({k:v for k,v in result.items() if k not in ('warped','ocr_image','variants','detection_overlay')},ensure_ascii=False,indent=2),encoding='utf-8')
        rows.append(row)
        print('[OK]' if result['success'] else '[FAIL]',filename,row['error'],row['score'])
    keys=['image','success','error','source','score','candidate_count','iou_gt']
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    print('성공:',sum(r['success'] for r in rows),'/',len(rows),'\n결과:',OUT)

if __name__=='__main__':main()

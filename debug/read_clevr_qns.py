import json
import os

clevr_qns = json.load(open('/hdd/robik/VQACP/filtered_vqa2/train_questions.json'))
main_qns = json.load(open('/hdd/robik/VQACP/questions/train_questions.json'))
print(f"clevr {len(clevr_qns['questions'])}")
print(f"main {len(main_qns)}")
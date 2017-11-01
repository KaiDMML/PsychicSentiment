from tqdm import tqdm
import json
pos2=0
neg2=0
pos3=0
neg3=0
all=[]
all_inp=[]
attack='darknetrelaunch_cratagged'
cnt=0
with open('all_sentiment_'+attack+'3.json','r') as f_input:
    for line in tqdm(f_input):
        cnt+=1
        tt_json = json.loads(line)
        print tt_json
        if int(tt_json['sentiment2'])==1:
            pos2+=1
        else:
            neg2+=1
        if int(tt_json['sentiment3'])==1:
            pos3+=1
        else:
            neg3+=1
        all_inp.append(tt_json['text'])
        all.append(tt_json)

print pos2,neg2,pos3,neg3,cnt,(pos2+neg2),(pos3+neg3)
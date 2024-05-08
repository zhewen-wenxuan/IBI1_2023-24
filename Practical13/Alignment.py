import blosum as bl
import os as os
with open('SLC6A4_HUMAN.fa','r') as human:
    human.content=human.read()
with open('SLC6A4_MOUSE.fa','r') as mouse:
    mouse.content=mouse.read()
with open('SLC6A4_RAT.fa','r') as rat:
    rat.content=rat.read()

human_str=''
mouse_str=''
rat_str=''

for line in human.content:
    if not line.startswith('<'):
        human_str+=line.replace('\n','')
for line in mouse.content:
    if not line.startswith('<'):
        mouse_str+=line.replace('\n','')
for line in rat.content:
    if not line.startswith('<'):
        rat_str+=line.replace('\n','')

matrix=bl.BLOSUM(62)

score_1=0, amino_1=0
score_2=0, amino_2=0
score_3=0, amino_3=0
for i in range(len(rat_str)):
    score+=matrix[]




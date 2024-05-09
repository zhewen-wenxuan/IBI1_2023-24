import blosum as bl
# with open('/Users/xuanzhewen/code/IBI1_2023-24/Practical13/SLC6A4_HUMAN.fa') as human:
human=open('/Users/xuanzhewen/code/IBI1_2023-24/Practical13/SLC6A4_HUMAN.fa','r')
with open('/Users/xuanzhewen/code/IBI1_2023-24/Practical13/SLC6A4_MOUSE.fa','r') as mouse:
    mouse.content=mouse.read()
with open('/Users/xuanzhewen/code/IBI1_2023-24/Practical13/SLC6A4_RAT.fa','r') as rat:
    rat.content=rat.read()

human_str=''
mouse_str=''
rat_str=''

for line in human:
    line = line.replace('\n','')
    if line.startswith('>'):
        human_str=""
    else:
        human_str += line.replace('\n','')
for line in mouse.content:
    if not line.startswith('>'):
        mouse_str+=line.replace('\n','')
for line in rat.content:
    if not line.startswith('>'):
        rat_str+=line.replace('\n','')

print(human)
# print(human_str)
# matrix=bl.BLOSUM(62)

# score_1=0 
# amino_1=0
# score_2=0 
# amino_2=0
# score_3=0 
# amino_3=0
# for i in range(len(rat_str)):
#     score_1+=matrix[human_str[i]][mouse_str[i]]
#     if human_str[i]==mouse_str[i]:
#         amino_1+=1
# for i in range(len(rat_str)):
#     score_2+=matrix[human_str[i]][rat_str[i]]
#     if human_str[i]==rat_str[i]:
#         amino_2+=1
# for i in range(len(rat_str)):
#     score_3+=matrix[rat_str[i]][mouse_str[i]]
#     if rat_str[i]==mouse_str[i]:
#         amino_3+=1
# print(score_1,amino_1,score_2,amino_2,score_3,amino_3)





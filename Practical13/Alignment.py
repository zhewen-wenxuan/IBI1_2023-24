import blosum as bl
human=open('/Users/xuanzhewen/code/IBI1_2023-24/Practical13/SLC6A4_HUMAN.fa','r')
mouse=open('/Users/xuanzhewen/code/IBI1_2023-24/Practical13/SLC6A4_MOUSE.fa','r')
rat=open('/Users/xuanzhewen/code/IBI1_2023-24/Practical13/SLC6A4_RAT.fa','r')

human_str=''
mouse_str=''
rat_str=''

human_str = ""  # 在循环外初始化 human_str
for line in human:
    line = line.strip()  # 去除行首和行尾的空格，包括 '\n'
    if line.startswith('>'):
        human_str = ""  # 如果是标题行，重置 human_str
    else:
        human_str += line  # 将行连接到 human_str
mouse_str=""
for line in mouse:
    line = line.strip()
    if line.startswith('>'):
        mouse_str =""
    else:
        mouse_str += line
rat_str = ""
for line in rat:
    line = line.strip()
    if line.startswith('>'):
        rat_str =""
    else:
        rat_str += line



print(human)
print(human_str)
matrix=bl.BLOSUM(62)

score_1=0 
amino_1=0
score_2=0 
amino_2=0
score_3=0 
amino_3=0
for i in range(len(rat_str)):
    score_1+=matrix[human_str[i]][mouse_str[i]]
    if human_str[i]==mouse_str[i]:
        amino_1+=1
for i in range(len(rat_str)):
    score_2+=matrix[human_str[i]][rat_str[i]]
    if human_str[i]==rat_str[i]:
        amino_2+=1
for i in range(len(rat_str)):
    score_3+=matrix[rat_str[i]][mouse_str[i]]
    if rat_str[i]==mouse_str[i]:
        amino_3+=1
print(score_1,amino_1,score_2,amino_2,score_3,amino_3)





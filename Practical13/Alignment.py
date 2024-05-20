import blosum as bl
human=open('/Users/xuanzhewen/code/IBI1_2023-24/Practical13/SLC6A4_HUMAN.fa','r')
mouse=open('/Users/xuanzhewen/code/IBI1_2023-24/Practical13/SLC6A4_MOUSE.fa','r')
rat=open('/Users/xuanzhewen/code/IBI1_2023-24/Practical13/SLC6A4_RAT.fa','r')

human_str=''
mouse_str=''
rat_str=''

human_str = ""#Initialize human_str outside of the loop
for line in human:
    line = line.strip()  # Remove spaces at the beginning and end of a line, including '\ n'
    if line.startswith('>'):
        human_str = ""  # If it is a title line, reset human_str
    else:
        human_str += line  # Connect rows to human_str
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
identical_percentage1=amino_1/len(rat_str)
print('BLOSUM score for human mouse comparison is ', score_1)
print('identity  for human mouse comparison is', identical_percentage1)
for i in range(len(rat_str)):
    score_2+=matrix[human_str[i]][rat_str[i]]
    if human_str[i]==rat_str[i]:
        amino_2+=1
identical_percentage2=amino_2/len(rat_str)
print('BLOSUM score for human rat comparison is ', score_2)
print('identity  for human rat comparison is', identical_percentage2)
for i in range(len(rat_str)):
    score_3+=matrix[rat_str[i]][mouse_str[i]]
    if rat_str[i]==mouse_str[i]:
        amino_3+=1
identical_percentage3=amino_3/len(rat_str)
print('BLOSUM score for rat mouse comparison is ', score_3)
print('identity  for rat mouse comparison is', identical_percentage3)

print('mouse sequence and rat sequence are most relative')
print('mouse is better than rat to be a organism model')
# mouse sequence and rat sequence are most relative
# mouse is better than rat to be a organism model   





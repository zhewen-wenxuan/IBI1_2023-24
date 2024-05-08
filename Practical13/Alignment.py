with open('SLC6A4_HUMAN.fa','r') as human:
    human.content=human.read()
with open('SLC6A4_MOUSE.fa','r') as mouse:
    mouse.content=mouse.read()
with open('SLC6A4_RAT.fa','r') as rat:
    rat.content=rat.read()
import BLOSUM as bl

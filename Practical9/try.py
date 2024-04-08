def remove_dups(name1, name2):
  L_rem = []
  L_copy = name1
  for i in L_copy:
    if i in name2:
      L_rem.append(i)
  for j in L_rem:
    L_copy.remove(j)
  return L_copy

List1=[1,2,3,4]
List2=[1,2,5,6]
al=remove_dups(name1=List1,name2=List2)
print(al)
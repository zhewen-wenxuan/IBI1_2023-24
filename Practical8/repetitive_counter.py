import re
def function_count(seq):
  List1=re.findall('GTGTGT',seq)
  List2=re.findall('GTCTGT',seq)
  count1=len(List1)
  count2=len(List2)
  total_count=count1+count2
  return total_count

seq='ATGCAATCGGTGTGTCTGTTCTGAGAGGGCCTAA'
answer=function_count(seq)
print(answer)
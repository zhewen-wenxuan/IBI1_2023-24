import re
input_file='/Users/xuanzhewen/code/IBI1_2023-24/Practical8/Saccharomyces_cerevisiae.R64-1-1.cdna.all.fa'
output_file='/Users/xuanzhewen/code/IBI1_2023-24/Practical8/duplicate_genes.fa'
pattern = re.compile(r'^>(\w+)_.*$')
flag=False #To choose the sequence only under duplication genes
with open(input_file,'r') as fin, open(output_file,'w') as fout:
    for line in fin:
        match = pattern.match(line)
        if match:
            if "duplication" in line:
                gene_name = match.group(1)  
                simplified_line = f">{gene_name}\n"  
                fout.write(simplified_line) 
                flag=True
            else:
                flag=False
        else:
            if flag:
                fout.write(line)
            
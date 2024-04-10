import re
input_file='/Users/xuanzhewen/code/IBI1_2023-24/Practical8/Saccharomyces_cerevisiae.R64-1-1.cdna.all.fa'
output_file='/Users/xuanzhewen/code/IBI1_2023-24/Practical8/duplicate_genes.fa'
pattern1 = re.compile(r'^>(\w+)_.*duplication.*$')
pattern2 = re.compile(r'^>(\w+)_.*(?!duplication).*$')
with open(input_file,'r') as fin, open(output_file,'w') as fout:
    for line in fin:
        match = pattern1.match(line)
        unmatch = pattern2.match(line)
        if match:
            gene_name = match.group(1)  
            simplified_line = f">{gene_name}\n"  
            fout.write(simplified_line) 
        else:
            fout.write(line)
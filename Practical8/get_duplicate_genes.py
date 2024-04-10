import re
input_file='Saccharomyces_cerevisiae.R64-1-1.cdna.all.fa'
output_file='duplicate_genes.fa'
pattern = re.compile(r'^>(\w+)_.*protein_coding.*$')
with open(input_file,'r') as fin, open(output_file,'w') as fout:
    for line in fin:
        match = pattern.match(line)
        if match:
            gene_name = match.group()  
            simplified_line = f">{gene_name}\n"  
            fout.write(simplified_line) 
        else:
            fout.write(line)
import re
user_repeat = input("'GTGTGT' or 'GTCTGT': ") 
input_file='/Users/xuanzhewen/code/IBI1_2023-24/Practical8/Saccharomyces_cerevisiae.R64-1-1.cdna.all.fa'
output_filename = f"{user_repeat}_duplicate_genes.fa"       
output_dir = '/Users/xuanzhewen/code/IBI1_2023-24/Practical8/'
output_file = output_dir + output_filename
with open(input_file, 'r') as fin:     
    lines = fin.readlines()
with open(output_filename, 'w') as fout:      
    gene = False             # Flag in the whole section, there only the first circulation that gene==false.
    current_gene_name = None
    current_gene_sequence = ""      

    for line in lines:                        #first jump the gene name and repeat times to record sequence,
        if line.startswith(">"):              #after record all sequence of one gene, then write in the gene name and repeat times
            if gene and current_gene_name:    #finally move to the next gene
                repeat_count = current_gene_sequence.count(user_repeat)     
                fout.write(f">{current_gene_name} \n")    
                fout.write(f"{repeat_count}  {current_gene_sequence} \n") 
            gene = True         
            current_gene_name = line.strip()[1:].split(' ')[0]     
            current_gene_sequence = ""
        else:
            if gene:                  
                current_gene_sequence += line.strip()        
print(f" {output_filename} is the file you want")
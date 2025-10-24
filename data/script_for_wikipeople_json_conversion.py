import sys
import os
import json

if len(sys.argv) < 3:
    exit()

input_path = sys.argv[1]
output_path = sys.argv[2]
print(f"input_path, {input_path} output_path={output_path}!")


if not os.path.exists(input_path):
    print("*** {} not found. Skipping. ***".format(input_path))
    exit()

lines = open(input_path, "r").read().splitlines()
with open(output_path, "w") as f:
    
    for i, line in enumerate(lines):                
        #print(f"Line = {i}>>>> {line}")
        record = json.loads(line)
        keys = list(record.keys())
        num_keys = len(keys)
        if num_keys < 3:
            continue
        head = keys[0] 
        tail =  keys[1]
        N = record[keys[2]]
        pr_tuple = list([head.partition('_')[0],record[head],record[tail]]) 
        #print(f"N={N}")
        qual_pairs=[]
        if N > 2:
            print(f"Line = {i}>>>> {line}")
            i = 3
            for k in range(3,num_keys):
                print(f"IN quals {k}")
                
                for arr_elt in record[keys[k]]:
                    print(f"arr elmts {arr_elt}")
                    qual_pairs.append(keys[k])
                    qual_pairs.append(arr_elt)

        print(f"primary tuple = {pr_tuple} qual pairs={qual_pairs}")
        f.write("<<<" + ",".join(map(str, pr_tuple)) + ">>>"+ ",".join(map(str, qual_pairs))+ "\n")
    
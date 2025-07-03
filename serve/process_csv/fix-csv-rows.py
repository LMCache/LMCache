import re
import csv
import sys

# 1) First pass: coalesce any line that starts with ']' into the end of the previous line.
#    (Adjust the regex if your broken lines start with something else.)
continuation_re = re.compile(r'^\]\s*')

input_path    = '/home/ubuntu/st-prodstack-v/LMCache/serve/results/Jun_19_1_coding/ours/05_old.csv'
staged_path   = '/home/ubuntu/st-prodstack-v/LMCache/serve/results/Jun_19_1_coding/ours/05-staged.csv'
output_path   = '/home/ubuntu/st-prodstack-v/LMCache/serve/results/Jun_19_1_coding/ours/05.csv'

with open(input_path,  'r', encoding='utf-8', errors='ignore') as fin, \
     open(staged_path, 'w', encoding='utf-8', newline='') as fout:
    
    prev = None
    for line in fin:
        if continuation_re.match(line):
            # merge this line onto the end of prev
            prev = prev.rstrip('\n') + line.lstrip()
        else:
            # flush the previous line (if any), then start a new one
            if prev is not None:
                fout.write(prev)
            prev = line
    # write the last buffered line
    if prev is not None:
        fout.write(prev)


# 2) Second pass: normalise to exactly N columns (15 in your case)
csv.field_size_limit(sys.maxsize)

with open(staged_path,  'r', encoding='utf-8', errors='ignore', newline='') as fin, \
     open(output_path,  'w', encoding='utf-8', newline='') as fout:

    reader = csv.reader(fin)
    writer = csv.writer(fout)

    # grab header, infer column count
    header   = next(reader)
    expected = len(header)
    writer.writerow(header)

    for row in reader:
        if len(row) < expected:
            # pad missing fields
            row += [''] * (expected - len(row))
        elif len(row) > expected:
            # shove any extra commas into the last field
            row = row[:expected-1] + [','.join(row[expected-1:])]
        writer.writerow(row)

import json

with open("./qanta.train.2018.04.18.json") as json_file:
    f = json.load(json_file)["questions"]
print(len(f))
print(f[0].keys())
f1 = open("./split_qanta_temp.json", "w")
#f_string=f[0].split("{\"text\"")
#print(len(f_string))
print(type(f))
f_s=f[:10]
with open("sample.json", "w") as outfile:
    json.dumps(f, indent=4)
with open("sample_short.json", "w") as outfile:
    json.dumps(f_s, indent=4)
f1.write("[")
for i in range (len(f)):
    json.dump(f[i],f1,indent=4)
    #f1.write(",")
    if i==10:
        break
    f1.write(",")
f1.write("]")
"""for i in range (len(f_string)):
    if i==0:
        f1.write(f_string[i])
    else:
        f1.write("{\"text\""+f_string[i]+"\n")
    if i==10:
        f1.write("{\"text\""+f_string[i]+"\n"+"]}")
        break"""
#print(f_string)
f1.close()

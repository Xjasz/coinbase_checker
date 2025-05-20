import re

log_file_path = "aaa.log"

print("start")

with open(log_file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

filtered_lines = []
for line in lines:
    if "-------------------------------------------------------" in line:
        continue
    if "USD:" in line:
        continue
    cleaned_line = re.sub(r"\([+-]\d+\)", "", line)
    filtered_lines.append(cleaned_line)

with open(log_file_path, "w", encoding="utf-8") as f:
    f.writelines(filtered_lines)



with open(log_file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
result = []
for line in lines:
    if "NEUTRAL SIGNALS:" in line:
        if result:
            result[-1] = result[-1].rstrip() + " " + line.strip() + "\n"
        continue
    result.append(line)
with open(log_file_path, "w", encoding="utf-8") as f:
    f.writelines(result)


print("stop")
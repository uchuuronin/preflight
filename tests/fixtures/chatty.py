import sys

line = "x" * 1023 + "\n"
for _ in range(60 * 1024):  # 60 MB of output
	sys.stdout.write(line)

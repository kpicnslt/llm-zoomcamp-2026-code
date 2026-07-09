
#!/bin/bash
# автор: https://t.me/i_odmin
echo "=== Топ-5 процессов по CPU ==="
ps -eo pid,ppid,cmd,%cpu --sort=-%cpu | head -n 6

echo ""
echo "=== Топ-5 процессов по RAM ==="
ps -eo pid,ppid,cmd,%mem --sort=-%mem | head -n 6

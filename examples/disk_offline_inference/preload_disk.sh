DISK_PATH=local_disk/
for file in $DISK_PATH/*; do
  cat "$file" > /dev/null
done

### kills screens matching certain format
screen -ls  | egrep "^\s*[0-9.a-Z_]+_C=+[0-9.a-Z_}{]" | awk -F "." '{print $1}' | xargs kill 2> /dev/null

echo "killed previous screens"

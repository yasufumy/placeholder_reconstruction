# Example
# ./bin/extract_gameid.sh ./dataset/F24 ./dataset/gameid.txt

#ls $1 -1 | awk '{print substr($0, 17, 6)}' > $2
ls $1 -1 | grep -o '[0-9]\{6\}' | uniq > $2

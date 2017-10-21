
import json

with open('./forum_example.json') as f_dark:
    for line in f_dark:
        json_data = json.loads(line)
        print json_data

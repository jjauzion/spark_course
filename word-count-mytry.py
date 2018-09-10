from pyspark import SparkConf, SparkContext
import re

def clean_word(word):
    match = re.findall('[a-zA-Z]', word)
    match = "".join(match)
    return match.lower()

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = conf)

input = sc.textFile("file:///Users/JJAUZION/Documents/dev/SparkCourse/book.txt")
words = input.flatMap(lambda x: x.split())
words = words.map(clean_word)
wordCounts = words.countByValue()
for word, count in wordCounts.items():
    cleanWord = word.encode('ascii', 'ignore')
    if (cleanWord):
        print(cleanWord.decode() + " " + str(count))
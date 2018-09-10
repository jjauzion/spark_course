from pyspark import SparkConf, SparkContext
import collections

def     parse_line(line):
    line = line.split()
    return (int(line[1]), 1)

conf = SparkConf().setMaster("local").setAppName("RatingsHistogram")
sc = SparkContext(conf = conf)

lines = sc.textFile("file:///Users/JJAUZION/Documents/dev/SparkCourse/data/ml-100k/u.data")
film_id = lines.map(parse_line)
film_id = film_id.reduceByKey(lambda x, y: x + y).sortBy(lambda x: x[1])


result = film_id.collect()
for film in result:
    print("{:4d}:{:5d} views".format(film[0], film[1]))

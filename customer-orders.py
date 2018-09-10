from pyspark import SparkConf, SparkContext

def     treat_csv_line(line):
    customer_id, item_id, price = line.split(',')
    return int(customer_id), float(price) 

conf = SparkConf().setMaster("local").setAppName("customer_orders")
sc = SparkContext(conf=conf)

customer_data = sc.textFile("file:///Users/JJAUZION/Documents/dev/SparkCourse/customer-orders.csv")
customer_data = customer_data.map(treat_csv_line).reduceByKey(lambda price1, price2: price1 + price2)
data_collected = customer_data.collect()
for item in data_collected:
    print("{:3d}:{:10.2f} $".format(item[0], item[1]))
from pyspark import SparkContext
from pyspark import SparkConf
import random
import urllib.request
import os

if __name__ == "__main__":

    conf = SparkConf().setMaster("local")
    sc = SparkContext(conf=conf)

    data = sc.textFile("TradeIn_Images.csv")
    header = data.first()
    data = data.filter(lambda line: line != header)

    rdd = data.map(lambda x: x.split(","))\
        .map(lambda x: (x[0] + ',' + x[1] + ',' + x[2] + ',' + x[3] + ',' + x[4] + ',' + x[5], x[6]))\
        .reduceByKey(lambda a, b: a + ',' + b)

    print(rdd.take(5))


    def download_image(url,num, autoId):

        img_name = num
        full_name = str(img_name) + '.jpg'

        if not os.path.exists(autoId):
            os.makedirs(autoId)
        fullfilename = os.path.join(autoId + '/', full_name)

        urllib.request.urlretrieve(url, fullfilename)

    image_list = ()

    for image in rdd.take(rdd.count()):
        image_list = image[1].split(",")

        print(len(image_list))

        num = 0
        for item in image_list:

            try:
                download_image(item, num, image[0])
<<<<<<< HEAD

=======
                
>>>>>>> origin/wayne-branch
            except: "404 error"

            num = num + 1
    sc.stop()
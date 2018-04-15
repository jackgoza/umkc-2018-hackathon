from label_image import inception_rank, inception_rank1
from pyspark import SparkContext
from pyspark import SparkConf

if __name__ == "__main__":

    tire_labels = "data/tire_output_labels.txt"
    tire_graph = "data/tire_output_graph.pb"
    damage_labels = "data/damage_output_labels.txt"
    damage_graph = "data/damage_output_graph.pb"
    interior_labels = "data/interior_output_labels.txt"
    interior_graph = "data/interior_output_graph.pb"
    rust_labels = "data/rust_output_labels.txt"
    rust_graph = "data/rust_output_graph.pb"
    category_labels = "data/category_output_labels.txt"
    category_graph = "data/category_output_graph.pb"

    conf = SparkConf().setMaster("local")
    sc = SparkContext(conf=conf)

    data = sc.textFile("TradeIn_Images.csv")
    header = data.first()
    data = data.filter(lambda line: line != header)

    rdd = data.map(lambda x: x.split(","))\
        .map(lambda x: (x[0] + ',' + x[1] + ',' + x[2] + ',' + x[3] + ',' + x[4] + ',' + x[5], x[6]))\
        .reduceByKey(lambda a, b: a + ',' + b)

    auto_id = "192358"
    rdd1 = rdd.filter(lambda x: auto_id in x[0])

    image_list = ()
    detail_list = ()

    for line in rdd1.take(rdd1.count()):
        detail_list = line[0].split(",")
        image_list = line[1].split(",")

        depreciated = float(detail_list[5])
        print("Maximum Allowed value is $"+str(depreciated))

        for item in image_list:

            try:
                category_rank = inception_rank(item, category_labels, category_graph)

                if category_rank[0][0] == 'body':
                    damage = inception_rank1(damage_labels, damage_graph)

                    if damage[0][0] == 'body_damage_yes':
                        depreciated = depreciated * 0.98

                    rust = inception_rank1(rust_labels, rust_graph)
                    if rust[0][0] == 'rust_yes':
                        depreciated = depreciated * 0.99

                if category_rank[0][0] == 'tire':
                    tire = inception_rank1(tire_labels, tire_graph)
                    if tire[0][0] == 'tire_tread_bad':
                        depreciated = depreciated * 0.98

                if category_rank[0][0] == 'interior':
                    interior = inception_rank1(interior_labels, interior_graph)
                    if interior[0][0] == 'interior_damage_yes':
                        depreciated = depreciated * 0.99

            except: "404 error"

        print("After Image Evaluation, Trade-In Price is $" + str(depreciated))


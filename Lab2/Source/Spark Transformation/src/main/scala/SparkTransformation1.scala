import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by nandanamudi on 2/1/17.
  */
object SparkTransformation1 {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Transformations").setMaster("local[1]")
    val sc = new SparkContext(conf)


    val l = sc.parallelize(List(1, 2, 3, 4, 5))
    val m = sc.parallelize(List(6, 7, 8, 9, 10))
    l.cartesian(m).collect.foreach(f => println(f))



    val x = sc.parallelize(List((1, "apple"), (2, "samsung"), (3, "hp"), (4, "asus")), 2)
    val y = sc.parallelize(List((1, "mobile"), (3, "printer"), (4, "laptop"), (1, "desktop"), (2, "iPad")), 2)
    x.cogroup(y).collect.foreach(f => println(f))
    println("Doing cogroup here")


    val dif = x.subtract(y)
    dif.collect().foreach(f => println(f._2))



    val n = sc.parallelize(List(1, 2, 1, 3), 1)
    val nn = sc.parallelize(List(5, 6, 5, 7), 1)
    val dd = n.zip(nn)
    dd.collectAsMap.foreach(f => println(f))
    println("collecting as Map here")


    val a1 = sc.parallelize(List("banana", "orange", "apple", "lemon", "grape", "cherry", "mango", "pear", "lemon"), 3)
    val b1 = sc.parallelize(List(1, 1, 2, 2, 2, 1, 2, 2, 2), 3)
    val c1 = b1.zip(a1)
    val d1 = c1.combineByKey(List(_), (l: List[String], m: String) => m :: l, (l: List[String], m: List[String]) => l ::: m)
    d1.collect.foreach(f => println(f))
    println("combining by Key here")



    val randRDD = sc.parallelize(List((2, "dog"), (6, "cat"), (7, "lion"), (3, "deer"), (4, "horse"), (1, "bear"), (5, "bird")), 3)
    val sortedRDD = randRDD.sortByKey()

    sortedRDD.filterByRange(1, 3).collect.foreach(f => println(f))
    println("filtering By Range here")
  }

}


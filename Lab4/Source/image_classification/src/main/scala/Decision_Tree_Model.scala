import java.nio.file.{Files, Paths}

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.mllib.tree.model.DecisionTreeModel

import scala.collection.mutable

/**
  * Created by JyothiKiran on 2/16/2017.
  */
object Decision_Tree_Model {

  def generateDecisionTreeModel(sc: SparkContext): Unit = {
    if (Files.exists(Paths.get(Pathsettings.DECISION_TREE_PATH))) {
      println(s"${Pathsettings.DECISION_TREE_PATH} exists, skipping Decision Tree model formation..")
      return
    }

    val data = sc.textFile(Pathsettings.HISTOGRAM_PATH)
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }

    // Split data into training (60%) and test (40%).
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    // Train a Decision Tree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 5
    val categoricalFeaturesInfo = Map[Int, Int]()
    //    val numTrees = 10 // Use more in practice.
    //    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    //    val impurity = "gini"
    //    val maxDepth = 4
    val maxBins = 100

    val numOfTrees = 5 to(10, 1)
    val strategies = List("all", "sqrt", "log2", "onethird")
    val maxDepths = 3 to(6, 1)
    val impurities = List("gini", "entropy")

    var bestModel: Option[DecisionTreeModel] = None
    var bestErr = 1.0
    val bestParams = new mutable.HashMap[Any, Any]()
    var bestnumTrees = 0
    var bestFeatureSubSet = ""
    var bestimpurity = ""
    var bestmaxdepth = 0


    impurities.foreach(impurity => {
      maxDepths.foreach(maxDepth => {

        println(" impurity " + impurity + " maxDepth " + maxDepth)

        val model = DecisionTree.trainClassifier(training, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

        val predictionAndLabel = test.map { point =>
          val prediction = model.predict(point.features)
          (point.label, prediction)
        }

        val testErr = predictionAndLabel.filter(r => r._1 != r._2).count.toDouble / test.count()
        println("Test Error = " + testErr)
        Evaluation_Confusion_matrix.evaluateModel(predictionAndLabel)

        if (testErr < bestErr) {
          bestErr = testErr
          bestModel = Some(model)

          bestParams.put("impurity", impurity)
          bestParams.put("maxDepth", maxDepth)
          bestimpurity = impurity

          bestmaxdepth = maxDepth
        }
      })
    })

println("Best Err " + bestErr)
println("Best params " + bestParams.toArray.mkString(" "))


val DecisionTreeModel = DecisionTree.trainClassifier(parsedData, numClasses, categoricalFeaturesInfo, bestimpurity, bestmaxdepth, maxBins)


// Save and load model
    DecisionTreeModel.save(sc, Pathsettings.DECISION_TREE_PATH)
println("-------------Decision Tree Model generated---------------")
}

}
